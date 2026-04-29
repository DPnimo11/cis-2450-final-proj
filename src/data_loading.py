import polars as pl

from src.config import HYBRID_TARGET_DATASET_PATH, MODELING_DATASET_PATH, RAW_MERGED_DATA_PATH


def load_merged_data(path=RAW_MERGED_DATA_PATH) -> pl.DataFrame:
    """Load the merged Bluesky/Yahoo Finance dataset with parsed timestamps."""
    df = pl.read_csv(path)

    if df.schema.get("Timestamp") == pl.String:
        df = df.with_columns(
            pl.col("Timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z")
        )

    return df


def load_modeling_data(path=MODELING_DATASET_PATH) -> pl.DataFrame:
    """Load the processed ticker-hour modeling dataset."""
    df = pl.read_csv(path)

    if df.schema.get("Timestamp") == pl.String:
        df = df.with_columns(
            pl.col("Timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z")
        )

    return df


def load_hybrid_target_data(path=HYBRID_TARGET_DATASET_PATH) -> pl.DataFrame:
    """Load the processed hybrid target modeling dataset."""
    df = pl.read_csv(path)

    datetime_cols = [
        "Timestamp",
        "Signal_Start",
        "Signal_End",
        "Target_Timestamp",
        "Reference_Timestamp",
    ]
    for col in datetime_cols:
        if col in df.columns and df.schema.get(col) == pl.String:
            df = df.with_columns(
                pl.col(col).str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z")
            )

    return df


def summarize_data_quality(df: pl.DataFrame) -> dict[str, object]:
    """Return common audit values used in EDA markdown and diagnostics."""
    unique_ticker_hours = df.select(["Ticker", "Timestamp"]).unique().height
    duplicate_post_keys = df.height - df.select(["Ticker", "Timestamp", "Text"]).unique().height

    return {
        "shape": df.shape,
        "unique_ticker_hours": unique_ticker_hours,
        "duplicate_post_keys": duplicate_post_keys,
        "timestamp_min": df.select(pl.col("Timestamp").min()).item(),
        "timestamp_max": df.select(pl.col("Timestamp").max()).item(),
    }
