# Project Context

## What this project is
A CIS 2450 (Big Data Analytics) final project that merges **Bluesky social media sentiment** with **Yahoo Finance hourly stock data** to predict short-term price movements. The pipeline scrapes Bluesky posts mentioning stock cashtags (e.g. `$AAPL`), scores them with **FinBERT** (GPU-accelerated), joins them with hourly OHLCV data from Yahoo Finance, and then performs EDA and machine learning modeling in a Jupyter notebook. **Due April 30th, 11:59 PM EST.**

## Current state
- **Data collection (`data_collection.py`)**: Fully functional and running. Scrapes Bluesky via `atproto`, scores sentiment with ProsusAI/FinBERT, fetches hourly Yahoo Finance data, performs a left join on `(Ticker, Timestamp)`, forward/backward fills financial columns for off-market hours, and appends to a growing CSV. The current dataset has **~194,891 rows** (well above the 50k requirement).
- **Notebook organization**: The old combined `eda_and_modeling.ipynb` is still present as a backup. New work has been split into `notebooks/01_data_audit_and_eda.ipynb`, `notebooks/02_feature_engineering.ipynb`, and `notebooks/03_modeling_and_results.ipynb`, with shared helper functions in `src/`.
- **Step 1 clean modeling input**: The feature-engineering notebook filters to timestamps on/after **2024-06-01**, aggregates posts to one row per `(Ticker, Timestamp)`, and validates **54,572 ticker-hour rows**, **23 tickers**, **0 nulls**, and **0 duplicate ticker-hour keys**.
- **Step 2 hybrid target dataset**: The feature-engineering notebook builds a hybrid intraday/overnight target with a tunable **0.1% return threshold**, drops tiny neutral moves, and validates **13,325 labeled rows**: intraday down/up = **4,995 / 5,001**, overnight down/up = **1,587 / 1,742**.
- **Step 3 final feature dataset**: `data/processed/feature_dataset.csv` is the only processed CSV artifact kept for modeling. It contains **13,325 rows**, **70 columns**, **0 nulls**, and includes lag returns, volume anomaly, sentiment EMA/z-score, post-count anomaly, time features, target-type flags, and ticker indicators.
- **Step 4 modeling state**: `notebooks/03_modeling_and_results.ipynb` now uses `data/processed/feature_dataset.csv`, compares combined/intraday/overnight scopes, compares no resampling/random upsampling/SMOTE, trains Logistic Regression, Random Forest, and Histogram Gradient Boosting, tunes the two tree-based models with `RandomizedSearchCV`, and saves result tables plus selected model artifacts.
- **Best current test results**: combined model F1 **0.609**, intraday-only F1 **0.562**, overnight-only F1 **0.611**. ROC-AUC remains close to **0.50-0.53**, so the honest conclusion should be that sentiment features produce limited directional signal in this sample.
- **Raw data is stored** in `data/raw/merged_financial_sentiment_data.csv` (~60MB, gitignored). Processed modeling data is stored in `data/processed/`.

## Codebase map
- `data_collection.py` — Main data pipeline: scrapes Bluesky, runs FinBERT, fetches Yahoo Finance, merges, deduplicates, and saves CSV.
- `eda_and_modeling.ipynb` — Original combined notebook, kept as a backup while the split notebooks are developed.
- `notebooks/01_data_audit_and_eda.ipynb` — Dataset audit, null checks, ticker counts, timestamp coverage, and current EDA visuals.
- `notebooks/02_feature_engineering.ipynb` — Current baseline target construction plus ticker-hour aggregation scaffold.
- `notebooks/03_modeling_and_results.ipynb` — Final modeling workflow with combined/split model scopes, resampling comparison, tuning, metrics, confusion matrices, feature importance where available, and saved model artifacts.
- `src/` — Shared helper modules for config, data loading, feature engineering, modeling, evaluation, and plotting.
- `outputs/` — Generated figures, model artifacts, and report tables for dashboard/presentation reuse.
- `outputs/tables/model_base_comparison.csv` — Base model/resampling comparison.
- `outputs/tables/model_tuned_comparison.csv` — Tuned tree-model comparison.
- `outputs/tables/model_final_summary.csv` — Final selected test metrics by modeling scope.
- `outputs/models/best_*_model.joblib` — Selected model artifacts for dashboard use.
- `data/raw/merged_financial_sentiment_data.csv` — The raw merged dataset from data collection (gitignored, ~194k rows).
- `data/processed/feature_dataset.csv` — Final processed modeling dataset (gitignored CSV artifact, ~10MB).
- `.env` — Bluesky credentials (`BLUESKY_HANDLE`, `BLUESKY_PASSWORD`). **Do NOT commit.**
- `requirements.txt` — Python dependencies (pandas, polars, yfinance, atproto, transformers, torch, scikit-learn, etc.).
- `venv/` — Python 3.13 virtual environment (gitignored).
- `notes.txt` — Working notes and next-step reminders (gitignored).
- `spec.txt` — Course project specification (gitignored).
- `rubric.txt` — Detailed grading rubric (gitignored).
- `README.md` — Brief project updates and TODO list.
- `cis_2450_final_project_proposal.pdf` — Original project proposal.

## How to run it
```bash
# Setup
cd "c:\Users\dpnim\Documents\cis 2450\cis-2450-final-proj"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Data collection (requires .env with Bluesky creds, GPU recommended for FinBERT)
python data_collection.py

# EDA, feature engineering, and modeling
jupyter notebook notebooks/01_data_audit_and_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling_and_results.ipynb
```

## Architecture notes

### Data flow
1. **Bluesky scraping**: For each of 23 tickers, fetches up to 200 pages × 100 posts via `atproto` search API. Each post is scored by FinBERT (`positive - negative` score → `Sentiment` column). Timestamps are truncated to the hour (UTC).
2. **Yahoo Finance**: Fetches hourly OHLCV data per ticker. Timestamps are converted to UTC and truncated to the hour.
3. **Merge**: Left join `social_df` onto `finance_df` on `(Ticker, Timestamp)`. This keeps all posts including off-market/weekend ones.
4. **Gap handling**: Financial columns (`Open`, `High`, `Low`, `Close`, `Volume`) are forward-filled then backward-filled per ticker so that overnight/weekend posts inherit the last known market values.
5. **Deduplication**: On append, existing CSV is loaded and combined with new data; duplicates are removed on `(Ticker, Timestamp, Text)` keeping the latest.
6. **Target variable**: The final target dataset now uses a hybrid strategy:
   - Intraday rows use regular-session sentiment at hour `t` to predict the next observed same-day market bar.
   - Overnight rows aggregate off-market sentiment before the next observed market open and predict the gap from the previous observed market close to that open.
   - The direction target is thresholded: `1` for returns above `+0.1%`, `0` for returns below `-0.1%`, and tiny neutral moves are dropped.

### Key libraries
- **Polars** (not Pandas) — used throughout for data wrangling (course requirement).
- **ProsusAI/FinBERT** via HuggingFace `transformers.pipeline` — sentiment analysis, GPU-accelerated with `torch.cuda`.
- **scikit-learn** — modeling, scaling, RandomizedSearchCV, Logistic Regression, Random Forest, Histogram Gradient Boosting.
- **imbalanced-learn** — SMOTE comparison after train split.
- **matplotlib / seaborn** — EDA visualizations.
- **yfinance** — hourly stock data.
- **atproto** — Bluesky API client.

### Tickers tracked
```python
["$AAPL", "$NVDA", "$TSLA", "$GME", "$MSFT", "$AMZN", "$META", "$GOOGL",
 "$AMD", "$SMCI", "$PLTR", "$AMC", "$INTC", "$NFLX", "$COIN", "$MSTR",
 "$HOOD", "$BABA", "$SPY", "$QQQ", "$DJT", "$RDDT", "$ARM"]
```

## Known issues / bugs
- **Old baseline target was flawed**: The original post-level baseline target created severe class imbalance and overweighted high-post hours. This is now addressed in `notebooks/02_feature_engineering.ipynb` with ticker-hour aggregation and a hybrid intraday/overnight target.
- **Neutral sentiment dominance**: Even with FinBERT, the vast majority of sentiment scores cluster near zero (neutral). Raw sentiment alone is a weak predictor.
- **Raw data collection preserves one row per post** (not one row per hour). This is still true in raw data, but final modeling now uses aggregated ticker-hour and hybrid target rows.
- **Predictive signal is modest**: The current best models beat naive presentation metrics like recall/F1 in some scopes, but ROC-AUC is near random. This is a result to present honestly, not a code bug.

## Constraints / preferences
- **Must use Polars** for data wrangling (course topic).
- **Must apply 6+ course topics** for full marks. Currently covered: Polars, FinBERT/text modeling, supervised learning, time-series feature engineering, ensemble models, imbalance handling, and hyperparameter tuning.
- **SMOTE/upsampling/downsampling MUST be applied AFTER train-test split** — doing it before is a rubric penalty (-3 points for data leakage).
- **Scaling/PCA must also be applied AFTER the split** (fit on train only, transform on test).
- **Need 3 models minimum**: 1 baseline + 2 more complex (e.g., Random Forest / XGBoost / etc.).
- **Need an interactive dashboard** (e.g., Streamlit or Dash) for the final deliverable.
- **8–10 min recorded presentation** required with slides, no code on slides.
- Codebase should be modular, documented, and readable.
- Two data sources required (Bluesky + Yahoo Finance ✓).
- 50k+ rows after cleaning required (✓, currently ~194k).

## Recent changes
- Switched from VADER to **FinBERT** for sentiment analysis (much better for financial text).
- Added **GPU detection** and batch processing (`batch_size=16`) for FinBERT inference.
- Expanded ticker list from a few to **23 tickers**.
- Changed merge strategy to **left join + forward fill** to preserve posts outside trading hours (previously an inner join was dropping overnight data).
- Old commented-out version of `fetch_bluesky_posts` is still in `data_collection.py` wrapped in triple quotes.
- Added progress printing during Bluesky scraping (page-by-page indicators).
- Split the old combined notebook into rubric-aligned notebooks and added reusable `src/` helper modules.
- Completed Step 1 clean modeling input in memory: valid finance-window filtering plus ticker-hour aggregation.
- Reorganized local CSV artifacts: raw collection output now lives under `data/raw/`, final processed modeling input under `data/processed/`, and `outputs/` is reserved for figures, models, and report tables.
- Completed Step 2 hybrid target construction with tunable thresholding and intraday/overnight target types.
- Completed Step 3 final feature dataset saved to `data/processed/feature_dataset.csv`; superseded intermediate processed CSVs were removed.
- Completed Step 4 modeling: combined plus split models, resampling strategy comparison, RandomizedSearchCV tuning, evaluation metrics, final summary tables, and saved model artifacts.

## TODO / next work
**Priority order for completing the project before April 30:**

1. **Fix target variable / class imbalance** (CRITICAL):
   - Step 1 and Step 2 are complete inside `notebooks/02_feature_engineering.ipynb`.
   - Step 3 is complete: use `data/processed/feature_dataset.csv` as the modeling input.
   - The hybrid target substantially improves the class balance versus the old post-level target.

2. **Feature engineering** (DONE in notebook 02):
   - Added sentiment EMA features, sentiment z-score, bullishness index, post-count anomaly, lag returns, volume anomaly, time features, target-type flags, and ticker indicators.
   - `Target_Return`, `Target_Price`, and other target columns remain in the final CSV for evaluation/explanation but must be excluded from training features.

3. **Modeling pipeline** (DONE in notebook 03):
   - SMOTE/random upsampling are applied only after the split and only to training data.
   - Scaling is fit on train only.
   - Trained Logistic Regression, Random Forest, and Histogram Gradient Boosting.
   - Tuned tree-based models with RandomizedSearchCV.
   - Evaluation includes Precision, Recall, F1, ROC-AUC, PR-AUC, and confusion matrices.
   - Feature importance is shown for selected Random Forest models where available.

4. **EDA polish**:
   - Need 3–5 excellent, well-formatted charts with markdown explanations for the presentation.
   - Consider using **Plotly** for interactive charts (bonus course topic).

5. **Dashboard**:
   - Build an interactive **Streamlit or Dash** dashboard.
   - Snapshot best model weights with `joblib` or `pickle` to load without retraining.

6. **Presentation**:
   - 8–10 min recorded presentation (slides, no code, all members speak).

## Gotchas
- **`data_collection.py` does NOT need to be re-run** for any feature engineering or modeling changes. All advanced data prep (EMA, aggregation, target variable redesign, SMOTE) should happen in the notebook.
- **EMA/rolling averages must be calculated on sentiment data BEFORE joining with financial data.** If you calculate them after a forward-fill, stale sentiment gets artificially amplified (e.g., one Friday post gets treated as if it was posted every hour all weekend).
- **Yahoo Finance only serves hourly data for the last 730 days.** The pipeline already clamps `min_date` to 700 days ago to stay within this window.
- **Timestamp precision matters**: Both social and financial timestamps are cast to `pl.Datetime("us", "UTC")` after truncation to `1h`. If you change this, the join will silently produce no matches.
- **`.env` contains real Bluesky app password** — never commit it. It's in `.gitignore`.
- **The old fetch function** is still in `data_collection.py` wrapped in triple quotes (lines 31–95). It's dead code — can be cleaned up.
- **`requirements.txt`** still lists `nltk` from the old VADER approach. It's not used anymore but won't hurt.
- **The notebook uses Polars for loading/wrangling** but casts to Pandas for seaborn/matplotlib plots and for scikit-learn. This is expected.
