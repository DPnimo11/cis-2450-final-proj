# Project Context

## What this project is
A CIS 2450 (Big Data Analytics) final project that merges **Bluesky social media sentiment** with **Yahoo Finance hourly stock data** to predict short-term price movements. The pipeline scrapes Bluesky posts mentioning stock cashtags (e.g. `$AAPL`), scores them with **FinBERT** (GPU-accelerated), joins them with hourly OHLCV data from Yahoo Finance, and then performs EDA and machine learning modeling in a Jupyter notebook. **Due April 30th, 11:59 PM EST.**

## Current state
- **Data collection (`data_collection.py`)**: Fully functional and running. Scrapes Bluesky via `atproto`, scores sentiment with ProsusAI/FinBERT, fetches hourly Yahoo Finance data, performs a left join on `(Ticker, Timestamp)`, forward/backward fills financial columns for off-market hours, and appends to a growing CSV. The current dataset has **~194,891 rows** (well above the 50k requirement).
- **Notebook organization**: The old combined `eda_and_modeling.ipynb` is still present as a backup. New work has been split into `notebooks/01_data_audit_and_eda.ipynb`, `notebooks/02_feature_engineering.ipynb`, and `notebooks/03_modeling_and_results.ipynb`, with shared helper functions in `src/`.
- **Step 1 clean modeling input**: `data/processed/modeling_dataset.csv` has been generated from the raw CSV. It filters to timestamps on/after **2024-06-01**, aggregates posts to one row per `(Ticker, Timestamp)`, and contains **54,572 ticker-hour rows**, **23 tickers**, **0 nulls**, and **0 duplicate ticker-hour keys**.
- **Current modeling state**: The split notebooks preserve the initial EDA and **baseline Logistic Regression** model with `class_weight='balanced'`. The baseline achieves ~0.74 ROC-AUC but only ~40% accuracy due to severe class imbalance. **No advanced models, SMOTE, hyperparameter tuning, final feature engineering, or dashboard have been implemented yet.**
- **Raw data is stored** in `data/raw/merged_financial_sentiment_data.csv` (~60MB, gitignored). Processed modeling data is stored in `data/processed/`.

## Codebase map
- `data_collection.py` — Main data pipeline: scrapes Bluesky, runs FinBERT, fetches Yahoo Finance, merges, deduplicates, and saves CSV.
- `eda_and_modeling.ipynb` — Original combined notebook, kept as a backup while the split notebooks are developed.
- `notebooks/01_data_audit_and_eda.ipynb` — Dataset audit, null checks, ticker counts, timestamp coverage, and current EDA visuals.
- `notebooks/02_feature_engineering.ipynb` — Current baseline target construction plus ticker-hour aggregation scaffold.
- `notebooks/03_modeling_and_results.ipynb` — Current baseline Logistic Regression workflow and evaluation scaffold.
- `src/` — Shared helper modules for config, data loading, feature engineering, modeling, evaluation, and plotting.
- `outputs/` — Generated figures, model artifacts, and report tables for dashboard/presentation reuse.
- `data/raw/merged_financial_sentiment_data.csv` — The raw merged dataset from data collection (gitignored, ~194k rows).
- `data/processed/modeling_dataset.csv` — Step 1 clean targetless modeling input (gitignored CSV artifact, ~12MB).
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
6. **Target variable**: The current baseline target is binary — `1` if the next close for the same ticker is above the current close, `0` otherwise. This target is still flawed and will be replaced in the feature-engineering notebook.

### Key libraries
- **Polars** (not Pandas) — used throughout for data wrangling (course requirement).
- **ProsusAI/FinBERT** via HuggingFace `transformers.pipeline` — sentiment analysis, GPU-accelerated with `torch.cuda`.
- **scikit-learn** — modeling (LogisticRegression baseline so far, StandardScaler).
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
- **Severe class imbalance**: Only ~5% of rows are labeled "Up" (`Close > Open`). This is primarily because overnight and weekend hours have the same `Close` value (forward-filled), so those rows always map to "Down/Flat." The baseline model gets 93% recall on "Up" but only 7% precision, with 40% overall accuracy.
- **Target variable is flawed**: Using `Close > Open` on the same forward-filled row doesn't capture real price movement. Overnight and off-market rows have identical `Open`/`Close` (both forward-filled from the same source), flooding the target with "Down" labels.
- **Neutral sentiment dominance**: Even with FinBERT, the vast majority of sentiment scores cluster near zero (neutral). Raw sentiment alone is a weak predictor.
- **Data collection preserves one row per post** (not one row per hour). This means viral hours with many tweets get disproportionately many rows, all sharing the same financial target. This distorts the model.

## Constraints / preferences
- **Must use Polars** for data wrangling (course topic).
- **Must apply 6+ course topics** for full marks. Currently covered: Polars, FinBERT (text/LLMs). Still need: supervised learning models (beyond baseline), hyperparameter tuning, time series features, ensemble models, and potentially more.
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
- Completed Step 1 clean modeling input: valid finance-window filtering plus ticker-hour aggregation saved to `data/processed/modeling_dataset.csv`.
- Reorganized local CSV artifacts: raw collection output now lives under `data/raw/`, processed modeling input under `data/processed/`, and `outputs/` is reserved for figures, models, and report tables.

## TODO / next work
**Priority order for completing the project before April 30:**

1. **Fix target variable / class imbalance** (CRITICAL):
   - Step 1 is complete: use `data/processed/modeling_dataset.csv` as the clean ticker-hour input.
   - Implement the **Hybrid Model** strategy in the notebook:
     - *Overnight*: Aggregate all sentiment from 4 PM to 9:30 AM into one row → predict the "morning gap" (Next Open - Previous Close).
     - *Intraday*: Pool all posts within each hour into one row → use sentiment at hour `t` to predict return at hour `t+1`.
   - This eliminates the flood of flat/down rows from off-market hours.

2. **Feature engineering** (in notebook, NOT in data_collection.py):
   - Resample sentiment to a continuous hourly timeline (fill missing hours with 0).
   - Calculate **EMA** of sentiment (short ~4h, long ~24h) — must be done BEFORE joining financial data.
   - Consider **Sentiment Z-Score** (standardize relative to a rolling window) to amplify rare spikes.
   - Consider a **Bullishness Index**: `(#positive - #negative) / #total` per hour.
   - `Post_Count` per hour is already available as a feature.
   - **Financial Features** to complement sentiment:
     - **Past 1-hour Return**: `(Close - Open) / Open` for the current hour.
     - **Past 24-hour Return**: Trailing 24-hour stock performance.
     - **Financial Volume Anomaly**: Standardized/Z-score of trading volume to detect unusual market activity.
     - **Hour of Day**: Integer (0-23) to capture intraday seasonality.

3. **Modeling pipeline** (in notebook):
   - Apply **SMOTE** (or up/downsampling) AFTER train-test split.
   - Scale features AFTER split (fit on train, transform on test).
   - Train **3 models**: Logistic Regression (baseline), then 2 more (e.g., Random Forest, XGBoost).
   - **Hyperparameter tuning**: Use Randomized Search or Bayesian Optimization (not just Grid Search).
   - **Evaluation**: Use Precision, Recall, F1, ROC-AUC (NOT just accuracy — rubric penalty for imbalanced data).
   - **Feature importance**: Extract and plot from the best tree-based model.

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
