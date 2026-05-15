#!/usr/bin/env python3
"""Train comment toxicity classification models."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from comment_toxicity import (
    Config,
    apply_target_encoding,
    build_features,
    build_post_features,
    build_preprocessor,
    evaluate_models,
    generate_submission,
    load_config,
    load_data,
    save_model,
    setup_logging,
    train_models,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train comment toxicity classifier")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--train-csv", help="Override train CSV path")
    parser.add_argument("--test-csv", help="Override test CSV path")
    parser.add_argument("--out-dir", help="Override output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)

    if args.train_csv:
        config.data.train_csv = args.train_csv
    if args.test_csv:
        config.data.test_csv = args.test_csv
    if args.out_dir:
        config.output.out_dir = args.out_dir

    train_df, test_df, _ = load_data(
        config.data.train_csv, config.data.test_csv, config.data.sample_csv
    )

    logger.info("Building features...")
    train_df = build_features(train_df)
    test_df = build_features(test_df)

    logger.info("Building post-level features...")
    train_df, test_df = build_post_features(train_df, test_df)

    logger.info("Applying target encoding...")
    te_cfg = config.preprocessing.target_encoding
    train_df, test_df, encoded_cols = apply_target_encoding(
        train_df, test_df, te_cfg.smoothing, te_cfg.n_folds, config.training.random_seed
    )

    logger.info("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(config, encoded_cols)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    logger.info("Starting model training...")
    results = train_models(X_train, y_train, test_df, preprocessor, config)

    logger.info("\n%s", evaluate_models(results, y_train).to_string(index=False))

    oof_proba = results["oof_proba"]
    blend_preds = (results["blend_weights"][0] * results["oof_lgb"] +
                   results["blend_weights"][1] * results["oof_lr"]).argmax(axis=1)
    logger.info("\nBlended OOF performance:\n%s", classification_report(y_train, blend_preds, digits=4))

    from comment_toxicity.train import predict_with_thresholds
    tuned_preds = predict_with_thresholds(oof_proba, results["thresholds"])
    logger.info("\nThreshold-tuned OOF performance:\n%s", classification_report(y_train, tuned_preds, digits=4))

    save_model(results, config)

    sub_path = str(Path(config.output.out_dir) / config.output.submission_name)
    generate_submission(results["test_proba"], results["thresholds"], test_df, sub_path)
    logger.info("Done! Submission saved to %s", sub_path)


if __name__ == "__main__":
    main()
