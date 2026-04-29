# CIS 2450 Final Project

This project merges Bluesky social sentiment with Yahoo Finance hourly stock data to study whether social sentiment helps predict short-term stock price movement.

## Repository Structure

```text
.
├── data_collection.py
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_and_results.ipynb
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── plots.py
├── outputs/
│   ├── figures/
│   ├── models/
│   └── tables/
└── data/
    └── merged_financial_sentiment_data.csv
```

The original `eda_and_modeling.ipynb` is still present as a backup. New work should happen in the `notebooks/` and `src/` structure.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Collection

Create a `.env` file with Bluesky credentials before running collection:

```text
BLUESKY_HANDLE=your-handle.bsky.social
BLUESKY_PASSWORD=your-app-password
```

Then run:

```bash
python data_collection.py
```

The collected CSV is stored at `data/merged_financial_sentiment_data.csv`. The data directory is gitignored.

## Notebook Order

1. `notebooks/01_data_audit_and_eda.ipynb`
   - Loads the merged dataset.
   - Checks shape, nulls, ticker balance, duplicates, and timestamp coverage.
   - Produces the current EDA visuals.

2. `notebooks/02_feature_engineering.ipynb`
   - Reproduces the current baseline target.
   - Aggregates post-level data to ticker-hour rows.
   - Saves an intermediate hourly modeling table to `outputs/tables/`.

3. `notebooks/03_modeling_and_results.ipynb`
   - Reproduces the current Logistic Regression baseline.
   - Uses chronological train/test split and train-only scaling.
   - Leaves space for the final model comparison and tuning work.

## Current Highest-Priority Work

- Replace the post-level target with the cleaned hourly/hybrid target.
- Filter timestamps to valid Yahoo Finance hourly coverage.
- Add feature engineering: lag returns, volume anomaly, sentiment EMA/z-score, hour/day features, and ticker encoding.
- Train and tune at least three models: Logistic Regression, Random Forest, and a boosting model.
- Build the required interactive dashboard.
- Prepare the 8-10 minute final presentation and slide PDF.
