"""Target encoding and preprocessing pipeline.

Implements out-of-fold smooth target encoding to prevent leakage,
and builds the full sklearn ColumnTransformer with TF-IDF, SVD,
categorical one-hot encoding, and numerical scaling.
"""

import copy
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from comment_toxicity.config import Config

logger = logging.getLogger(__name__)

TE_COLS = [
    "race",
    "religion",
    "gender",
    "disability_str",
    "race_gender",
    "race_religion",
    "gender_religion",
    "demo_signature",
    "demo_missing_pattern",
]
CLASSES = [0, 1, 2, 3]


def apply_target_encoding(
    train_df: pd.DataFrame, test_df: pd.DataFrame, smoothing: float, n_folds: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply out-of-fold smooth mean target encoding for demographic columns.

    Uses StratifiedKFold to compute target means without leakage.
    Smoothing pulls rare categories toward the global mean.

    Returns:
        Tuple of (train_df, test_df, list of new encoded column names).
    """
    n = len(train_df)
    global_rate = {c: (train_df["label"] == c).sum() / n for c in CLASSES}

    te_train: dict[str, np.ndarray] = {}
    for col in TE_COLS:
        for c in CLASSES:
            te_train[f"{col}_p{c}"] = np.full(n, global_rate[c], dtype=np.float32)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df, train_df["label"]):
        tr_fold = train_df.iloc[tr_idx]
        val_fold = train_df.iloc[val_idx]
        for col in TE_COLS:
            for c in CLASSES:
                cname = f"{col}_p{c}"
                stats = (
                    tr_fold.assign(_t=(tr_fold["label"] == c).astype(int))
                    .groupby(col, observed=True)["_t"]
                    .agg(["sum", "count"])
                )
                rates = (stats["sum"] + smoothing * global_rate[c]) / (stats["count"] + smoothing)
                te_train[cname][val_idx] = (
                    val_fold[col].map(rates).fillna(global_rate[c]).values
                )

    for cname, arr in te_train.items():
        train_df[cname] = arr

    for col in TE_COLS:
        for c in CLASSES:
            cname = f"{col}_p{c}"
            stats = (
                train_df.assign(_t=(train_df["label"] == c).astype(int))
                .groupby(col, observed=True)["_t"]
                .agg(["sum", "count"])
            )
            rates = (stats["sum"] + smoothing * global_rate[c]) / (stats["count"] + smoothing)
            test_df[cname] = test_df[col].map(rates).fillna(global_rate[c]).values

    encoded_cols = [f"{col}_p{c}" for col in TE_COLS for c in CLASSES]
    logger.info("Target encoding done, train shape: %s", train_df.shape)
    return train_df, test_df, encoded_cols


def build_preprocessor(config: Config, encoded_cols: list[str]) -> ColumnTransformer:
    """Build the full preprocessing pipeline.

    Transformers applied in parallel:
    - Word TF-IDF (1,2) grams
    - Word TF-IDF via SVD (50 dims)
    - Char TF-IDF (3,5) grams
    - Categorical one-hot encoding (race, religion, gender)
    - Numerical StandardScaler
    """
    categorical_cols = ["race", "religion", "gender"]

    numerical_cols = _get_numerical_cols(encoded_cols)

    cfg = config.preprocessing

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    num_pipeline = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(cfg.tfidf_word.ngram_range),
                    min_df=cfg.tfidf_word.min_df,
                    max_df=cfg.tfidf_word.max_df,
                    max_features=cfg.tfidf_word.max_features,
                    sublinear_tf=cfg.tfidf_word.sublinear_tf,
                ),
                "comment_clean",
            ),
            (
                "word_svd",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=tuple(cfg.tfidf_svd.ngram_range),
                                min_df=cfg.tfidf_svd.min_df,
                                max_df=cfg.tfidf_svd.max_df,
                                max_features=cfg.tfidf_svd.max_features,
                                sublinear_tf=cfg.tfidf_svd.sublinear_tf,
                            ),
                        ),
                        ("svd", TruncatedSVD(n_components=cfg.tfidf_svd.n_components, random_state=42)),
                    ]
                ),
                "comment_clean",
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(cfg.tfidf_char.ngram_range),
                    min_df=cfg.tfidf_char.min_df,
                    max_df=cfg.tfidf_char.max_df,
                    max_features=cfg.tfidf_char.max_features,
                    sublinear_tf=cfg.tfidf_char.sublinear_tf,
                ),
                "comment_clean",
            ),
            ("categorical", cat_pipeline, categorical_cols),
            ("numerical", num_pipeline, numerical_cols),
        ],
        remainder="drop",
        n_jobs=-1,
    )

    logger.info(
        "Preprocessor built: %d numerical, %d categorical features",
        len(numerical_cols),
        len(categorical_cols),
    )
    return preprocessor


def fit_transform_fold(
    preprocessor: ColumnTransformer, X_tr: pd.DataFrame, X_va: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Deep-copy preprocessor, fit on train, transform all splits.

    Args:
        preprocessor: Base (unfit) ColumnTransformer.
        X_tr: Training features.
        X_va: Validation features.
        X_test: Test features.

    Returns:
        Transformed arrays (X_tr, X_va, X_test).
    """
    fold_prep = copy.deepcopy(preprocessor)
    fold_prep.fit(X_tr)
    X_tr_proc = fold_prep.transform(X_tr)
    X_va_proc = fold_prep.transform(X_va)
    X_test_proc = fold_prep.transform(X_test)
    return X_tr_proc, X_va_proc, X_test_proc


def _get_numerical_cols(encoded_cols: list[str]) -> list[str]:
    return [
        "emoticon_1",
        "emoticon_2",
        "emoticon_3",
        "upvote",
        "downvote",
        "if_1",
        "if_2",
        "char_count",
        "word_count",
        "comment_hour",
        "comment_dayofweek",
        "comment_month",
        "comment_year",
        "url_count",
        "has_url",
        "quote_count",
        "ellipsis_count",
        "repeated_punct_count",
        "all_caps_word_count",
        "max_word_len",
        "digit_token_count",
        "symbol_token_count",
        "long_word_count",
        "short_word_ratio",
        "sentence_count",
        "avg_sentence_len",
        "std_word_len",
        "repeated_char_token_count",
        "mixed_char_token_count",
        "vote_ratio",
        "vote_total",
        "emoticon_total",
        "has_emoticon",
        "excl_count",
        "quest_count",
        "caps_ratio",
        "unique_word_ratio",
        "avg_word_len",
        "demo_count",
        "has_any_demo",
        "race_missing",
        "religion_missing",
        "gender_missing",
        "all_demo_missing",
        "if_1_log",
        "if_2_log",
        "upvote_log",
        "downvote_log",
        "if1_if2_ratio",
        "if1_gt0",
        "disability_enc",
        "post_count",
        "post_avg_upvote",
        "post_avg_downvote",
        "post_avg_if1",
        "post_avg_if2",
        "post_avg_char_count",
        "post_label_0_rate",
        "post_label_1_rate",
        "post_label_2_rate",
        "post_label_3_rate",
        "post_comment_rank",
        "post_comment_position",
        "seconds_since_prev_comment",
        "seconds_since_first_comment",
        "upvote_vs_post_avg",
        "downvote_vs_post_avg",
        "if1_vs_post_avg",
        "if2_vs_post_avg",
        "char_count_vs_post_avg",
    ] + encoded_cols
