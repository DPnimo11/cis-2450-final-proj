import polars as pl

from src.config import BASELINE_FEATURES


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
                pl.len().alias("Post_Count"),
            ]
        )
        .sort(["Ticker", "Timestamp"])
    )
