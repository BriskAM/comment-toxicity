"""Inference module for generating predictions and submissions."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from comment_toxicity.train import predict_with_thresholds

logger = logging.getLogger(__name__)


def generate_submission(
    test_proba: np.ndarray,
    thresholds: np.ndarray,
    test_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Generate a Kaggle-format submission CSV.

    Args:
        test_proba: (n_samples, 4) probability array from ensemble.
        thresholds: Per-class decision thresholds.
        test_df: Test dataframe (used for ID column if present).
        output_path: Path to save submission.csv.

    Returns:
        Submission dataframe with ID and label columns.
    """
    test_preds = predict_with_thresholds(test_proba, thresholds)

    ids = test_df["ID"] if "ID" in test_df.columns else pd.RangeIndex(1, len(test_preds) + 1)

    submission = pd.DataFrame({"ID": ids, "label": test_preds})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info(
        "Saved submission to %s, shape: %s, distribution:\n%s",
        output_path,
        submission.shape,
        submission["label"].value_counts().sort_index().to_string(),
    )
    return submission
