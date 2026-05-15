# Comment Toxicity Classification

Project site: <https://briskam.github.io/comment-toxicity/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BriskAM/comment-toxicity/blob/main/notebooks/original_analysis.ipynb)
[![CI](https://github.com/BriskAM/comment-toxicity/actions/workflows/ci.yml/badge.svg)](https://github.com/BriskAM/comment-toxicity/actions/workflows/ci.yml)

Multi-class comment toxicity classifier (labels 0-3) using LightGBM ensemble with rich feature engineering, out-of-fold target encoding, per-class threshold tuning, and Nelder-Mead blend optimization.

## Results

Measured via 2-fold stratified CV OOF predictions on 198K training samples.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **LightGBM (blend + tuned)** | 0.912 | **0.817** | 0.913 |
| LightGBM | 0.913 | 0.813 | 0.913 |
| Random Forest (subsampled) | 0.825 | 0.587 | 0.823 |
| Logistic Regression | 0.725 | 0.494 | 0.698 |

## Highlights

- **LightGBM** primary model with class weights (`{0:1.0, 1:4.0, 2:2.0, 3:8.0}`) for severe class imbalance.
- **66+ engineered features**: text stats, punctuation/style, datetime parts, engagement ratios, demographic flags.
- **Post-level features**: per-thread label rates, comment position, time deltas between comments.
- **OOF target encoding**: smooth mean encoding for demographic features via 5-fold StratifiedKFold to prevent leakage.
- **Pipeline**: sklearn `ColumnTransformer` with word TF-IDF (1,2-grams), char TF-IDF (3,5-grams), SVD-reduced TF-IDF, OHE for categoricals, StandardScaler for numericals.
- **Nelder-Mead optimization** for ensemble blend weights and per-class decision thresholds.
- **Config-driven**: all hyperparameters in YAML with dataclass validation.

## Repository Layout

```
configs/default.yaml        Training and inference configuration
src/comment_toxicity/       Reusable Python package
  config.py                 YAML config + dataclass models
  data.py                   Data loading
  features.py               Feature engineering (vectorized)
  preprocessing.py          TF-IDF pipeline + target encoding
  train.py                  Model training, blend, threshold tuning
  inference.py              Prediction and submission generation
scripts/train.py            CLI training entrypoint
scripts/infer.py            CLI inference entrypoint
notebooks/                  Original Kaggle notebook archive
tests/                      pytest smoke tests
```

## Data Layout

Expected CSV format (matching the Kaggle competition):

```text
data/
  train.csv          (15 columns: created_date, post_id, emoticon_1..3, upvote,
                      downvote, if_1, if_2, race, religion, gender, disability,
                      comment, label)
  test.csv           (14 columns, same as above minus label)
  Sample.csv         (optional, for ID alignment)
```

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```sh
python scripts/train.py --config configs/default.yaml
```

With custom data paths:

```sh
python scripts/train.py \
  --config configs/default.yaml \
  --train-csv /path/to/train.csv \
  --test-csv /path/to/test.csv \
  --out-dir runs/toxicity
```

## Inference

Generate a Kaggle-format submission CSV:

```sh
python scripts/infer.py \
  --config configs/default.yaml \
  --ensemble runs/comment_toxicity_ensemble.npz \
  --train-csv data/train.csv \
  --test-csv data/test.csv \
  --output submission.csv
```

## Docker

```sh
docker build -t comment-toxicity .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" comment-toxicity
```

## Run Tests

```sh
pytest tests/ -v
```

## Quick Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BriskAM/comment-toxicity/blob/main/notebooks/original_analysis.ipynb)

The notebook shows the full EDA, feature engineering, model comparison, and hyperparameter tuning used to develop this pipeline.

## Resume Bullet

Built **Comment Toxicity**, a production-grade toxicity classification pipeline using **LightGBM ensemble** with stratified CV, **OOF target encoding**, **Nelder-Mead blend optimization**, and per-class threshold tuning, achieving **0.817 macro F1** on a heavily imbalanced 198K-sample 4-class dataset.
