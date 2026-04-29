from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

RAW_MERGED_DATA_PATH = RAW_DATA_DIR / "merged_financial_sentiment_data.csv"
MODELING_DATASET_PATH = PROCESSED_DATA_DIR / "modeling_dataset.csv"
HYBRID_TARGET_DATASET_PATH = PROCESSED_DATA_DIR / "hybrid_target_dataset.csv"

# Yahoo Finance hourly data is limited to recent history. The raw Bluesky search
# can return much older posts, so final modeling starts after the finance window.
CLEAN_FINANCE_START = datetime(2024, 6, 1, tzinfo=timezone.utc)
MARKET_TIME_ZONE = "America/New_York"
MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 16
TARGET_RETURN_THRESHOLD = 0.001
MAX_INTRADAY_HORIZON_HOURS = 6
MAX_OVERNIGHT_HORIZON_HOURS = 96

RANDOM_STATE = 42
TEST_SIZE = 0.20

TICKERS = [
    "$AAPL",
    "$NVDA",
    "$TSLA",
    "$GME",
    "$MSFT",
    "$AMZN",
    "$META",
    "$GOOGL",
    "$AMD",
    "$SMCI",
    "$PLTR",
    "$AMC",
    "$INTC",
    "$NFLX",
    "$COIN",
    "$MSTR",
    "$HOOD",
    "$BABA",
    "$SPY",
    "$QQQ",
    "$DJT",
    "$RDDT",
    "$ARM",
]

BASELINE_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Sentiment",
    "Post_Count",
]
