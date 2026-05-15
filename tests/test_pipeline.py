"""Smoke tests for the comment toxicity package."""

import numpy as np
import pandas as pd
import pytest

from comment_toxicity import (
    Config,
    build_features,
    build_post_features,
    apply_target_encoding,
    build_preprocessor,
    predict_with_thresholds,
)


@pytest.fixture
def sample_raw_df():
    """Create a minimal training dataframe."""
    n = 100
    return pd.DataFrame({
        "created_date": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "post_id": np.random.randint(1, 6, n),
        "emoticon_1": np.random.randint(0, 3, n),
        "emoticon_2": np.random.randint(0, 3, n),
        "emoticon_3": np.random.randint(0, 3, n),
        "upvote": np.random.randint(0, 10, n),
        "downvote": np.random.randint(0, 5, n),
        "if_1": np.random.randint(0, 20, n),
        "if_2": np.random.randint(0, 15, n),
        "race": np.random.choice(["white", "black", "asian", np.nan], n),
        "religion": np.random.choice(["christian", "muslim", np.nan], n),
        "gender": np.random.choice(["male", "female", np.nan], n),
        "disability": np.random.choice([True, False], n),
        "comment": ["test comment number " + str(i) for i in range(n)],
        "label": np.random.choice([0, 1, 2, 3], n, p=[0.57, 0.08, 0.32, 0.03]),
    })


@pytest.fixture
def sample_test_df():
    n = 50
    return pd.DataFrame({
        "created_date": pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC"),
        "post_id": np.random.randint(1, 6, n),
        "emoticon_1": np.random.randint(0, 3, n),
        "emoticon_2": np.random.randint(0, 3, n),
        "emoticon_3": np.random.randint(0, 3, n),
        "upvote": np.random.randint(0, 10, n),
        "downvote": np.random.randint(0, 5, n),
        "if_1": np.random.randint(0, 20, n),
        "if_2": np.random.randint(0, 15, n),
        "race": np.random.choice(["white", "black", np.nan], n),
        "religion": np.random.choice(["christian", np.nan], n),
        "gender": np.random.choice(["male", "female", np.nan], n),
        "disability": np.random.choice([True, False], n),
        "comment": ["example comment " + str(i) for i in range(n)],
    })


def test_build_features(sample_raw_df):
    df = build_features(sample_raw_df)
    assert df.shape[0] == 100
    assert df.shape[1] > 40
    assert "char_count" in df.columns
    assert "word_count" in df.columns
    assert "vote_ratio" in df.columns
    assert "demo_count" in df.columns
    assert df["word_count"].dtype in (np.int64, np.int32, int)


def test_build_post_features(sample_raw_df, sample_test_df):
    train = build_features(sample_raw_df)
    test = build_features(sample_test_df)
    train_out, test_out = build_post_features(train, test)
    assert "post_count" in train_out.columns
    assert "post_avg_upvote" in train_out.columns
    assert "upvote_vs_post_avg" in train_out.columns
    assert len(train_out) == 100
    assert len(test_out) == 50


def test_target_encoding(sample_raw_df, sample_test_df):
    train = build_features(sample_raw_df)
    test = build_features(sample_test_df)
    train, test = build_post_features(train, test)

    train_out, test_out, encoded_cols = apply_target_encoding(
        train, test, smoothing=20.0, n_folds=3, seed=42
    )
    assert len(encoded_cols) > 0
    assert (0 <= train_out[encoded_cols[0]]).all() and (train_out[encoded_cols[0]] <= 1).all()


def test_build_preprocessor(sample_raw_df, sample_test_df):
    train = build_features(sample_raw_df)
    test = build_features(sample_test_df)
    train, test = build_post_features(train, test)
    train, test, encoded_cols = apply_target_encoding(train, test, smoothing=20.0, n_folds=3, seed=42)

    config = Config()
    config.preprocessing.tfidf_word.min_df = 1
    config.preprocessing.tfidf_word.max_features = 50
    config.preprocessing.tfidf_svd.min_df = 1
    config.preprocessing.tfidf_svd.max_features = 50
    config.preprocessing.tfidf_char.min_df = 1
    config.preprocessing.tfidf_char.max_features = 50
    preprocessor = build_preprocessor(config, encoded_cols)

    result = preprocessor.fit_transform(train.drop(columns=["label"]))
    assert result.shape[0] == 100
    assert result.shape[1] > 50


def test_predict_with_thresholds():
    proba = np.array([[0.8, 0.1, 0.05, 0.05], [0.3, 0.5, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]])
    thresholds = np.array([1.0, 1.0, 1.0, 1.0])
    preds = predict_with_thresholds(proba, thresholds)
    assert (preds == [0, 1, 2]).all()

    thresholds2 = np.array([2.0, 1.0, 0.5, 1.0])
    preds2 = predict_with_thresholds(proba, thresholds2)
    assert preds2[0] == 0  # 0.8/2.0 = 0.4 > 0.05/0.5 = 0.1


def test_config_load():
    config = Config.from_yaml("configs/default.yaml")
    assert config.models.lightgbm.learning_rate == 0.03
    assert config.training.n_splits == 2
    assert config.preprocessing.tfidf_word.max_features == 8000
