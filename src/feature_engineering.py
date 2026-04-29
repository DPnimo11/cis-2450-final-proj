import polars as pl

from src.config import BASELINE_FEATURES, CLEAN_FINANCE_START


def add_next_close_target(
    df: pl.DataFrame,
    target_col: str = "Target_Direction",
) -> pl.DataFrame:
    """Add the current baseline target: whether the next close is above current close."""
    return (
        df.sort(["Ticker", "Timestamp"])
        .with_columns(pl.col("Close").shift(-1).over("Ticker").alias("Next_Close"))
        .drop_nulls(subset=["Next_Close"])
        .with_columns((pl.col("Next_Close") > pl.col("Close")).cast(pl.Int32).alias(target_col))
    )


def select_baseline_xy(
    df: pl.DataFrame,
    features: list[str] | None = None,
    target_col: str = "Target_Direction",
):
    """Select feature and target arrays for the current baseline model."""
    feature_cols = features or BASELINE_FEATURES
    X = df.select(feature_cols).to_pandas()
    y = df.select(target_col).to_numpy().ravel()
    return X, y


def filter_valid_finance_window(
    df: pl.DataFrame,
    start=CLEAN_FINANCE_START,
    end=None,
) -> pl.DataFrame:
    """Remove rows outside the reliable hourly Yahoo Finance coverage window."""
    filtered = (
        df.filter(pl.col("Timestamp") >= start)
        .drop_nulls(subset=["Open", "High", "Low", "Close", "Volume", "Sentiment"])
        .filter(pl.col("Volume") > 0)
    )

    if end is not None:
        filtered = filtered.filter(pl.col("Timestamp") <= end)

    return filtered.sort(["Ticker", "Timestamp"])


def aggregate_to_ticker_hour(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate post-level rows to one row per ticker-hour for the next modeling pass."""
    return (
        df.group_by(["Ticker", "Timestamp"])
        .agg(
            [
                pl.col("Open").first(),
                pl.col("High").first(),
                pl.col("Low").first(),
                pl.col("Close").first(),
                pl.col("Volume").first(),
                pl.col("Sentiment").mean().alias("Sentiment_Mean"),
                pl.col("Sentiment").median().alias("Sentiment_Median"),
                pl.col("Sentiment").std().fill_null(0).alias("Sentiment_Std"),
                pl.col("Sentiment").min().alias("Sentiment_Min"),
                pl.col("Sentiment").max().alias("Sentiment_Max"),
                (pl.col("Sentiment") > 0.05).sum().alias("Positive_Post_Count"),
                (pl.col("Sentiment") < -0.05).sum().alias("Negative_Post_Count"),
                (pl.col("Sentiment").abs() <= 0.05).sum().alias("Neutral_Post_Count"),
                pl.len().alias("Post_Count"),
            ]
        )
        .with_columns(
            (
                (pl.col("Positive_Post_Count") - pl.col("Negative_Post_Count"))
                / pl.col("Post_Count")
            ).alias("Bullishness_Index")
        )
        .sort(["Ticker", "Timestamp"])
    )


def build_clean_hourly_dataset(
    df: pl.DataFrame,
    start=CLEAN_FINANCE_START,
    end=None,
) -> pl.DataFrame:
    """Build the Step 1 clean modeling input: valid finance window plus ticker-hour rows."""
    valid_df = filter_valid_finance_window(df, start=start, end=end)
    return aggregate_to_ticker_hour(valid_df)
