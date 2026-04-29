# CIS 2450 Final Project

This project merges Bluesky social sentiment with Yahoo Finance hourly stock data to study whether social sentiment helps predict short-term stock price movement.

## Repository Structure

```text
.
├── data_collection.py
├── dashboard.py
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
   - Produces presentation-ready EDA visuals and saves them to `outputs/figures/`.

2. `notebooks/02_feature_engineering.ipynb`
   - Reproduces the current baseline target.
   - Filters to reliable Yahoo Finance hourly coverage starting `2024-06-01`.
   - Aggregates post-level data to one row per ticker-hour.
   - Builds the hybrid intraday/overnight target.
   - Adds final modeling features and saves only `data/processed/feature_dataset.csv`.

3. `notebooks/03_modeling_and_results.ipynb`
   - Loads `data/processed/feature_dataset.csv`.
   - Uses chronological train/test split and train-only scaling.
   - Compares combined, intraday-only, and overnight-only models.
   - Compares no resampling, random upsampling, and SMOTE.
   - Saves result tables to `outputs/tables/` and selected models to `outputs/models/`.

## Key Outputs

Presentation-ready EDA figures:

- `outputs/figures/eda_01_ticker_coverage.png`
- `outputs/figures/eda_02_sentiment_distribution.png`
- `outputs/figures/eda_03_monthly_activity_sentiment.png`
- `outputs/figures/eda_04_target_balance.png`
- `outputs/figures/eda_05_sentiment_vs_target.png`
- `outputs/figures/eda_06_example_ticker_timeline.png`

Model result tables:

- `outputs/tables/model_base_comparison.csv`
- `outputs/tables/model_tuned_comparison.csv`
- `outputs/tables/model_final_results.csv`
- `outputs/tables/model_final_summary.csv`

Saved model artifacts:

- `outputs/models/best_combined_model.joblib`
- `outputs/models/best_intraday_model.joblib`
- `outputs/models/best_overnight_model.joblib`

Current selected test results:

| Scope | Selected model | Resampling | F1 | ROC-AUC |
| --- | --- | --- | ---: | ---: |
| Combined | Logistic Regression | None | 0.609 | 0.533 |
| Intraday | Logistic Regression | None | 0.562 | 0.505 |
| Overnight | Tuned Random Forest | None | 0.611 | 0.513 |

The honest modeling conclusion is that sentiment and social activity provide a limited directional signal in this sample. F1/recall are usable for discussion, but ROC-AUC remains close to random, so the project should not overclaim predictive power.

## Current Project Status

- Step 1 complete: clean ticker-hour modeling input has `54,572` rows, `23` tickers, no nulls, and no duplicate ticker-hour keys.
- Step 2 complete: hybrid target dataset has `13,325` labeled rows with a tunable `0.1%` return threshold.
- Step 3 complete: final feature dataset has `13,325` rows and `70` columns.
- Step 4 complete: Logistic Regression, Random Forest, and Histogram Gradient Boosting have been trained/tuned across combined and split target scopes.
- EDA polish complete: the EDA notebook now tells the project story and saves reusable figures.
- Dashboard handoff: `dashboard.py` exists, but it should read the final `data/processed/feature_dataset.csv` rather than old `modeling_dataset.csv` paths.
- Prepare the 8-10 minute final presentation and slide PDF.

## Rubric Coverage Notes

- Two data sources: Bluesky social posts and Yahoo Finance hourly OHLCV.
- 50k+ rows after cleaning: the clean ticker-hour audit has `54,572` rows.
- EDA: notebook 01 includes data quality checks and six presentation-ready visuals.
- Preprocessing and feature engineering: notebook 02 handles filtering, ticker-hour aggregation, hybrid target construction, rolling features, lag returns, volume/post anomalies, time features, and ticker indicators.
- Modeling: notebook 03 includes a baseline model, two ensemble/advanced model families, chronological splits, train-only scaling, train-only resampling comparisons, hyperparameter tuning, and multiple metrics.
- Course topics used: Polars/data wrangling, text/sentiment modeling with FinBERT, supervised learning, time-series feature engineering, ensemble models, imbalance handling, hyperparameter tuning, Plotly/dashboard visualization.

## Run Checks

Use the notebooks in order:

```bash
source .venv/bin/activate
jupyter notebook notebooks/01_data_audit_and_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling_and_results.ipynb
```

For a terminal validation run:

```bash
JUPYTER_CONFIG_DIR=/tmp/jupyter_config \
JUPYTER_DATA_DIR=/tmp/jupyter_data \
JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime \
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit_and_eda.ipynb
```

Repeat the same command with notebooks 02 and 03 when regenerating the full pipeline.

Important submission note: `data/` is gitignored. Either include `data/raw/merged_financial_sentiment_data.csv` and `data/processed/feature_dataset.csv` in the final zip, or make sure the grader can regenerate them from the notebooks/scripts.
