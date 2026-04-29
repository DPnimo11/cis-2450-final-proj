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
    ├── raw/
    │   └── merged_financial_sentiment_data.csv
    └── processed/
        └── feature_dataset.csv
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

The collected CSV is stored at `data/raw/merged_financial_sentiment_data.csv`. The data directory is gitignored.

## Notebook Order

1. `notebooks/01_data_audit_and_eda.ipynb`
   - Loads the merged dataset.
   - Checks shape, nulls, ticker balance, duplicates, and timestamp coverage.
   - Produces the current EDA visuals.

2. `notebooks/02_feature_engineering.ipynb`
   - Reproduces the current baseline target.
   - Filters to reliable Yahoo Finance hourly coverage starting `2024-06-01`.
   - Aggregates post-level data to one row per ticker-hour.
   - Builds the hybrid intraday/overnight target.
   - Adds final modeling features and saves only `data/processed/feature_dataset.csv`.

3. `notebooks/03_modeling_and_results.ipynb`
   - Should now load `data/processed/feature_dataset.csv`.
   - Uses chronological train/test split and train-only scaling.
   - Leaves space for the final feature set, model comparison, and tuning work.

## Current Highest-Priority Work

- Step 1 complete: clean ticker-hour modeling input has `54,572` rows, `23` tickers, no nulls, and no duplicate ticker-hour keys.
- Step 2 complete: hybrid target dataset has `13,325` labeled rows with a tunable `0.1%` return threshold.
- Step 3 complete: final feature dataset has `13,325` rows and `70` columns.
- Train and tune at least three models: Logistic Regression, Random Forest, and a boosting model.
- Build the required interactive dashboard.
- Prepare the 8-10 minute final presentation and slide PDF.
