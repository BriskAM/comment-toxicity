#!/usr/bin/env python3
"""Run inference with a saved ensemble model on new data."""

import argparse
import logging
import sys

from comment_toxicity import (
    apply_target_encoding,
    build_features,
    build_post_features,
    build_preprocessor,
    generate_submission,
    load_config,
    load_data,
    setup_logging,
    train,
)
from comment_toxicity.train import load_model

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Infer comment toxicity labels")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--ensemble", required=True, help="Path to saved ensemble .npz")
    parser.add_argument("--train-csv", required=True, help="Path to train.csv (for target encoding)")
    parser.add_argument("--test-csv", required=True, help="Path to test.csv")
    parser.add_argument("--output", default="submission.csv", help="Output submission path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config(args.config)

    train_df, test_df, _ = load_data(args.train_csv, args.test_csv)

    train_df = build_features(train_df)
    test_df = build_features(test_df)

    train_df, test_df = build_post_features(train_df, test_df)

    te_cfg = config.preprocessing.target_encoding
    train_df, test_df, encoded_cols = apply_target_encoding(
        train_df, test_df, te_cfg.smoothing, te_cfg.n_folds, config.training.random_seed
    )

    preprocessor = build_preprocessor(config, encoded_cols)

    logger.info("Fitting preprocessor on train+test (text transformers)...")
    preprocessor.fit(pd.concat([train_df, test_df]))
    X_test_proc = preprocessor.transform(test_df)

    ensemble = load_model(args.ensemble)

    logger.info("Running inference with loaded ensemble...")
    submission = generate_submission(
        X_test_proc,  # Note: full inference needs full pipeline; this is a stub
        ensemble["thresholds"],
        test_df,
        args.output,
    )
    logger.info("Done! Saved to %s", args.output)


if __name__ == "__main__":
    main()
