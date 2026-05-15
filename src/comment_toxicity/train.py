"""Model training with StratifiedKFold CV, ensemble blending, and threshold tuning.

Trains LightGBM (primary), LogisticRegression (baseline), and
RandomForest (baseline) using 2-fold stratified cross-validation.
Optimizes blend weights and per-class thresholds via Nelder-Mead.
"""

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from comment_toxicity.config import Config

logger = logging.getLogger(__name__)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    preprocessor,
    config: Config,
) -> dict:
    """Train LightGBM, LogisticRegression, and RandomForest with 2-fold CV.

    Args:
        X_train: Training features (includes 'label' column when inside CV).
        y_train: Training labels (0-3).
        X_test: Test features.
        preprocessor: Unfit sklearn ColumnTransformer.
        config: Configuration dataclass.

    Returns:
        Dictionary with OOF predictions, test predictions, blend weights,
        thresholds, and training metadata.
    """
    cfg = config.training
    model_cfg = config.models
    n = len(X_train)

    oof_lgb = np.zeros((n, 4), dtype=np.float32)
    test_lgb = np.zeros((len(X_test), 4), dtype=np.float32)
    oof_lr = np.zeros((n, 4), dtype=np.float32)
    test_lr = np.zeros((len(X_test), 4), dtype=np.float32)
    oof_rf = np.zeros((n, 4), dtype=np.float32)
    test_rf = np.zeros((len(X_test), 4), dtype=np.float32)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_seed)

    from comment_toxicity.preprocessing import fit_transform_fold

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        t_start = time.time()
        logger.info("--- Fold %d ---", fold + 1)

        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        X_tr_proc, X_va_proc, X_test_proc = fit_transform_fold(
            preprocessor, X_tr, X_va, X_test
        )
        logger.info("Preprocessor fitted, shape: %s", X_tr_proc.shape)

        _train_lightgbm(
            X_tr_proc, y_tr, X_va_proc, y_va, X_test_proc,
            oof_lgb, test_lgb, va_idx, fold, model_cfg, cfg,
        )
        _train_logistic_regression(
            X_tr_proc, y_tr, X_va_proc, X_test_proc,
            oof_lr, test_lr, va_idx, fold, model_cfg, cfg,
        )
        _train_random_forest(
            X_tr_proc, y_tr, X_va_proc, X_test_proc,
            oof_rf, test_rf, va_idx, fold, model_cfg, cfg,
        )

        logger.info("Fold %d done in %.1fs", fold + 1, time.time() - t_start)

    blend_weights = _optimize_blend_weights(oof_lgb, oof_lr, y_train, cfg)
    oof_proba = blend_weights[0] * oof_lgb + blend_weights[1] * oof_lr
    thresholds = _optimize_thresholds(oof_proba, y_train, cfg)
    test_proba = blend_weights[0] * test_lgb + blend_weights[1] * test_lr

    return {
        "oof_lgb": oof_lgb,
        "oof_lr": oof_lr,
        "oof_rf": oof_rf,
        "test_lgb": test_lgb,
        "test_lr": test_lr,
        "test_rf": test_rf,
        "test_proba": test_proba,
        "blend_weights": blend_weights,
        "thresholds": thresholds,
        "oof_proba": oof_proba,
    }


def predict_with_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply per-class thresholds before argmax for calibrated predictions."""
    return (proba / thresholds.reshape(1, -1)).argmax(axis=1)


def _train_lightgbm(
    X_tr_proc, y_tr, X_va_proc, y_va, X_test_proc,
    oof, test_preds, va_idx, fold, model_cfg, cfg,
):
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=model_cfg.lightgbm.n_estimators,
        learning_rate=model_cfg.lightgbm.learning_rate,
        num_leaves=model_cfg.lightgbm.num_leaves,
        max_depth=model_cfg.lightgbm.max_depth,
        min_child_samples=model_cfg.lightgbm.min_child_samples,
        subsample=model_cfg.lightgbm.subsample,
        subsample_freq=model_cfg.lightgbm.subsample_freq,
        colsample_bytree=model_cfg.lightgbm.colsample_bytree,
        reg_alpha=model_cfg.lightgbm.reg_alpha,
        reg_lambda=model_cfg.lightgbm.reg_lambda,
        class_weight=model_cfg.class_weight,
        random_state=cfg.random_seed + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(
        X_tr_proc, y_tr,
        eval_set=[(X_va_proc, y_va)],
        callbacks=[
            lgb.early_stopping(model_cfg.lightgbm.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    oof[va_idx] = clf.predict_proba(X_va_proc)
    test_preds += clf.predict_proba(X_test_proc) / cfg.n_splits
    best_iter = clf.best_iteration_ or model_cfg.lightgbm.n_estimators
    logger.info("LightGBM done, best_iter=%d (%.1fs)", best_iter, time.time() - t0)


def _train_logistic_regression(
    X_tr_proc, y_tr, X_va_proc, X_test_proc,
    oof, test_preds, va_idx, fold, model_cfg, cfg,
):
    t0 = time.time()
    lr = LogisticRegression(
        C=model_cfg.logistic_regression.C,
        solver=model_cfg.logistic_regression.solver,
        max_iter=model_cfg.logistic_regression.max_iter,
        multi_class=model_cfg.logistic_regression.multi_class,
        class_weight=model_cfg.class_weight,
        n_jobs=-1,
        random_state=cfg.random_seed + fold,
    )
    lr.fit(X_tr_proc, y_tr)
    oof[va_idx] = lr.predict_proba(X_va_proc)
    test_preds += lr.predict_proba(X_test_proc) / cfg.n_splits
    logger.info("LogisticRegression done (%.1fs)", time.time() - t0)


def _train_random_forest(
    X_tr_proc, y_tr, X_va_proc, X_test_proc,
    oof, test_preds, va_idx, fold, model_cfg, cfg,
):
    t0 = time.time()
    n = X_tr_proc.shape[0]
    rng = np.random.RandomState(cfg.random_seed + fold)
    sample_idx = rng.choice(n, size=int(model_cfg.random_forest.subsample_ratio * n), replace=False)
    X_tr_rf = X_tr_proc[sample_idx] if sp.issparse(X_tr_proc) else X_tr_proc[sample_idx]
    y_tr_rf = y_tr.iloc[sample_idx]

    rf = RandomForestClassifier(
        n_estimators=model_cfg.random_forest.n_estimators,
        max_depth=model_cfg.random_forest.max_depth,
        max_features=model_cfg.random_forest.max_features,
        min_samples_leaf=model_cfg.random_forest.min_samples_leaf,
        class_weight=model_cfg.class_weight,
        random_state=cfg.random_seed + fold,
        n_jobs=-1,
    )
    rf.fit(X_tr_rf, y_tr_rf)
    oof[va_idx] = rf.predict_proba(X_va_proc)
    test_preds += rf.predict_proba(X_test_proc) / cfg.n_splits
    logger.info("RandomForest done (%.1fs)", time.time() - t0)


def _optimize_blend_weights(
    oof_lgb: np.ndarray,
    oof_lr: np.ndarray,
    y: pd.Series,
    cfg,
) -> np.ndarray:
    logger.info("Optimizing blend weights via %s...", cfg.blend_optimizer)

    def loss(weights):
        w = np.abs(weights) / np.sum(np.abs(weights))
        proba = w[0] * oof_lgb + w[1] * oof_lr
        return -f1_score(y, proba.argmax(axis=1), average="macro")

    res = minimize(loss, [0.5, 0.1], method=cfg.blend_optimizer, options={"maxiter": cfg.blend_maxiter})
    best_w = np.abs(res.x) / np.sum(np.abs(res.x))
    logger.info("Blend weights: lgb=%.3f, lr=%.3f", best_w[0], best_w[1])
    return best_w


def _optimize_thresholds(
    oof_proba: np.ndarray,
    y: pd.Series,
    cfg,
) -> np.ndarray:
    logger.info("Optimizing per-class thresholds via %s...", cfg.threshold_optimizer)

    def f1_loss(thresholds, proba, y_true):
        preds = predict_with_thresholds(proba, thresholds)
        return -f1_score(y_true, preds, average="macro")

    res = minimize(
        f1_loss, [1.0, 1.0, 1.0, 1.0],
        args=(oof_proba, y),
        method=cfg.threshold_optimizer,
        options={"maxiter": cfg.threshold_maxiter},
    )
    logger.info("Tuned thresholds: %s", np.round(res.x, 4).tolist())
    return res.x


def evaluate_models(results: dict, y_train: pd.Series) -> pd.DataFrame:
    """Compute accuracy and F1 scores for all models using OOF predictions."""
    from sklearn.metrics import accuracy_score

    rows = []
    for name, oof in [
        ("LightGBM", results["oof_lgb"]),
        ("LogisticRegression", results["oof_lr"]),
        ("RandomForest", results["oof_rf"]),
    ]:
        preds = oof.argmax(axis=1)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_train, preds),
            "macro_f1": f1_score(y_train, preds, average="macro"),
            "weighted_f1": f1_score(y_train, preds, average="weighted"),
        })
    return pd.DataFrame(rows)


def save_model(results: dict, config: Config) -> str:
    """Save ensemble artifacts (blend weights, thresholds) as numpy archives."""
    out_dir = Path(config.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{config.output.model_name}_ensemble.npz"
    np.savez_compressed(
        str(path),
        blend_weights=results["blend_weights"],
        thresholds=results["thresholds"],
    )
    logger.info("Saved ensemble to %s", path)
    return str(path)


def load_model(ensemble_path: str) -> dict:
    """Load saved blend weights and thresholds."""
    data = np.load(ensemble_path)
    return {
        "blend_weights": data["blend_weights"],
        "thresholds": data["thresholds"],
    }
