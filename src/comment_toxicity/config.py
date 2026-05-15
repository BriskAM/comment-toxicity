import logging
import logging.config
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging with timestamps and module names."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, stream=sys.stdout)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


@dataclass
class DataConfig:
    train_csv: str = "data/train.csv"
    test_csv: str = "data/test.csv"
    sample_csv: str = "data/Sample.csv"


@dataclass
class TfidfConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 10
    max_df: float = 0.95
    max_features: int = 8000
    sublinear_tf: bool = True


@dataclass
class TfidfSvdConfig(TfidfConfig):
    n_components: int = 50


@dataclass
class TargetEncodingConfig:
    smoothing: float = 20.0
    n_folds: int = 5


@dataclass
class PreprocessingConfig:
    tfidf_word: TfidfConfig = field(default_factory=TfidfConfig)
    tfidf_svd: TfidfSvdConfig = field(default_factory=TfidfSvdConfig)
    tfidf_char: TfidfConfig = field(default_factory=lambda: TfidfConfig(ngram_range=(3, 5), max_features=5000))
    target_encoding: TargetEncodingConfig = field(default_factory=TargetEncodingConfig)


@dataclass
class LightGBMConfig:
    n_estimators: int = 1200
    learning_rate: float = 0.03
    num_leaves: int = 127
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.6
    reg_alpha: float = 0.05
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 30


@dataclass
class LogisticRegressionConfig:
    C: float = 4.0
    solver: str = "saga"
    max_iter: int = 200
    multi_class: str = "multinomial"


@dataclass
class RandomForestConfig:
    n_estimators: int = 15
    max_depth: int = 5
    max_features: str = "sqrt"
    min_samples_leaf: int = 10
    subsample_ratio: float = 0.3


@dataclass
class ModelConfig:
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    logistic_regression: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    class_weight: dict[str, float] = field(default_factory=lambda: {"0": 1.0, "1": 4.0, "2": 2.0, "3": 8.0})


@dataclass
class TrainingConfig:
    n_splits: int = 2
    random_seed: int = 42
    blend_optimizer: str = "Nelder-Mead"
    blend_maxiter: int = 500
    threshold_optimizer: str = "Nelder-Mead"
    threshold_maxiter: int = 500
    tune_subsample: float = 0.25


@dataclass
class OutputConfig:
    out_dir: str = "runs"
    model_name: str = "comment_toxicity"
    submission_name: str = "submission.csv"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "Config":
        data_cfg = DataConfig(**d.get("data", {}))
        prep = d.get("preprocessing", {})
        prep_cfg = PreprocessingConfig(
            tfidf_word=TfidfConfig(**prep.get("tfidf_word", {})),
            tfidf_svd=TfidfSvdConfig(**prep.get("tfidf_svd", {})),
            tfidf_char=TfidfConfig(**prep.get("tfidf_char", {})),
            target_encoding=TargetEncodingConfig(**prep.get("target_encoding", {})),
        )
        models = d.get("models", {})
        models_cfg = ModelConfig(
            lightgbm=LightGBMConfig(**models.get("lightgbm", {})),
            logistic_regression=LogisticRegressionConfig(**models.get("logistic_regression", {})),
            random_forest=RandomForestConfig(**models.get("random_forest", {})),
            class_weight=models.get("class_weight", {"0": 1.0, "1": 4.0, "2": 2.0, "3": 8.0}),
        )
        training_cfg = TrainingConfig(**d.get("training", {}))
        output_cfg = OutputConfig(**d.get("output", {}))
        return cls(data=data_cfg, preprocessing=prep_cfg, models=models_cfg, training=training_cfg, output=output_cfg)


def load_config(config_path: str) -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = Config.from_yaml(str(path))
    logger.info("Loaded config from %s", config_path)
    return config
