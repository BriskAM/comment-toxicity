"""Data loading utilities."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: str, sample_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load train, test, and optional sample submission CSVs.

    Args:
        train_path: Path to train.csv.
        test_path: Path to test.csv.
        sample_path: Path to Sample.csv (optional).

    Returns:
        Tuple of (train_df, test_df, sample_df or None).
    """
    for path_str, name in [(train_path, "train"), (test_path, "test")]:
        if not Path(path_str).exists():
            raise FileNotFoundError(f"{name} data not found at {path_str}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_path) if sample_path and Path(sample_path).exists() else None

    logger.info(
        "Loaded data: train=%s, test=%s",
        train_df.shape,
        test_df.shape,
    )
    return train_df, test_df, sample_df
