import warnings

import polars as pl

from src.config import (
    BASELINE_FEATURES,
    CLEAN_FINANCE_START,
    MARKET_CLOSE_HOUR,
    MARKET_OPEN_HOUR,
    MARKET_TIME_ZONE,
    MAX_INTRADAY_HORIZON_HOURS,
    MAX_OVERNIGHT_HORIZON_HOURS,
    TARGET_RETURN_THRESHOLD,
)


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
                (pl.col("Sentiment") > 0.05).cast(pl.Int64).sum().alias("Positive_Post_Count"),
                (pl.col("Sentiment") < -0.05).cast(pl.Int64).sum().alias("Negative_Post_Count"),
                (pl.col("Sentiment").abs() <= 0.05).cast(pl.Int64).sum().alias("Neutral_Post_Count"),
                pl.len().alias("Post_Count"),
            ]
        )
        .with_columns(
            (
                (
                    pl.col("Positive_Post_Count").cast(pl.Float64)
                    - pl.col("Negative_Post_Count").cast(pl.Float64)
                )
                / pl.col("Post_Count").cast(pl.Float64)
            ).alias("Bullishness_Index")
        )
        .sort(["Ticker", "Timestamp"])
    )


def recompute_bullishness_index(df: pl.DataFrame) -> pl.DataFrame:
    """Recompute bullishness from signed post-count columns."""
    count_cols = ["Positive_Post_Count", "Negative_Post_Count", "Neutral_Post_Count", "Post_Count"]
    if not all(col in df.columns for col in count_cols):
        return df

    return df.with_columns(
        [pl.col(col).cast(pl.Int64) for col in count_cols]
    ).with_columns(
        (
            (
                pl.col("Positive_Post_Count").cast(pl.Float64)
                - pl.col("Negative_Post_Count").cast(pl.Float64)
            )
            / pl.col("Post_Count").cast(pl.Float64)
        ).alias("Bullishness_Index")
    )


def build_clean_hourly_dataset(
    df: pl.DataFrame,
    start=CLEAN_FINANCE_START,
    end=None,
) -> pl.DataFrame:
    """Build the Step 1 clean modeling input: valid finance window plus ticker-hour rows."""
    valid_df = filter_valid_finance_window(df, start=start, end=end)
    return aggregate_to_ticker_hour(valid_df)


def add_market_time_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add New York market calendar fields and a coarse regular-session flag."""
    local_ts = pl.col("Timestamp").dt.convert_time_zone(MARKET_TIME_ZONE)
    return df.with_columns(
        [
            local_ts.alias("Market_Timestamp"),
            local_ts.dt.date().alias("Market_Date"),
            local_ts.dt.weekday().alias("Market_Weekday"),
            local_ts.dt.hour().alias("Market_Hour"),
        ]
    ).with_columns(
        (
            (pl.col("Market_Weekday") <= 5)
            & (pl.col("Market_Hour") >= MARKET_OPEN_HOUR)
            & (pl.col("Market_Hour") < MARKET_CLOSE_HOUR)
        ).alias("Is_Market_Hour")
    )


def add_thresholded_direction_target(
    df: pl.DataFrame,
    threshold: float = TARGET_RETURN_THRESHOLD,
    return_col: str = "Target_Return",
    target_col: str = "Target_Direction",
    drop_neutral: bool = True,
) -> pl.DataFrame:
    """Convert returns to a tunable binary target and optionally drop tiny moves."""
    labeled = df.with_columns(
        pl.when(pl.col(return_col) > threshold)
        .then(pl.lit(1))
        .when(pl.col(return_col) < -threshold)
        .then(pl.lit(0))
        .otherwise(None)
        .cast(pl.Int8)
        .alias(target_col)
    )

    if drop_neutral:
        labeled = labeled.drop_nulls(subset=[target_col])

    return labeled


def _target_columns() -> list[str]:
    return [
        "Target_Type",
        "Ticker",
        "Timestamp",
        "Signal_Start",
        "Signal_End",
        "Target_Timestamp",
        "Reference_Timestamp",
        "Target_Horizon_Hours",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Sentiment_Mean",
        "Sentiment_Median",
        "Sentiment_Std",
        "Sentiment_Min",
        "Sentiment_Max",
        "Positive_Post_Count",
        "Negative_Post_Count",
        "Neutral_Post_Count",
        "Post_Count",
        "Bullishness_Index",
        "Reference_Price",
        "Target_Price",
        "Target_Return",
        "Target_Direction",
    ]


def build_intraday_target_rows(
    hourly_df: pl.DataFrame,
    threshold: float = TARGET_RETURN_THRESHOLD,
    max_horizon_hours: int = MAX_INTRADAY_HORIZON_HOURS,
    drop_neutral: bool = True,
) -> pl.DataFrame:
    """Build intraday rows that use market-hour sentiment to predict the next same-day bar."""
    hourly_df = recompute_bullishness_index(hourly_df)
    market_rows = (
        add_market_time_columns(hourly_df)
        .filter(pl.col("Is_Market_Hour"))
        .sort(["Ticker", "Timestamp"])
    )

    intraday = (
        market_rows.with_columns(
            [
                pl.col("Timestamp").shift(-1).over("Ticker").alias("Target_Timestamp"),
                pl.col("Close").shift(-1).over("Ticker").alias("Target_Price"),
                pl.col("Market_Date").shift(-1).over("Ticker").alias("Target_Market_Date"),
            ]
        )
        .drop_nulls(subset=["Target_Timestamp", "Target_Price"])
        .filter(pl.col("Target_Market_Date") == pl.col("Market_Date"))
        .with_columns(
            [
                pl.lit("intraday").alias("Target_Type"),
                pl.col("Timestamp").alias("Signal_Start"),
                pl.col("Timestamp").alias("Signal_End"),
                pl.col("Timestamp").alias("Reference_Timestamp"),
                pl.col("Close").alias("Reference_Price"),
                (
                    (pl.col("Target_Timestamp") - pl.col("Timestamp"))
                    .dt.total_seconds()
                    .cast(pl.Float64)
                    / 3600.0
                ).alias("Target_Horizon_Hours"),
                ((pl.col("Target_Price") - pl.col("Close")) / pl.col("Close")).alias(
                    "Target_Return"
                ),
            ]
        )
        .filter(pl.col("Target_Horizon_Hours") <= max_horizon_hours)
    )

    intraday = add_thresholded_direction_target(
        intraday, threshold=threshold, drop_neutral=drop_neutral
    )
    return intraday.select(_target_columns())


def build_overnight_target_rows(
    hourly_df: pl.DataFrame,
    threshold: float = TARGET_RETURN_THRESHOLD,
    max_horizon_hours: int = MAX_OVERNIGHT_HORIZON_HOURS,
    drop_neutral: bool = True,
) -> pl.DataFrame:
    """Aggregate off-market sentiment before the next observed market open."""
    hourly_df = recompute_bullishness_index(hourly_df)
    with_market_time = add_market_time_columns(hourly_df).sort(["Ticker", "Timestamp"])
    market_rows = with_market_time.filter(pl.col("Is_Market_Hour"))

    daily_market = (
        market_rows.sort(["Ticker", "Timestamp"])
        .group_by(["Ticker", "Market_Date"], maintain_order=True)
        .agg(
            [
                pl.col("Timestamp").first().alias("Target_Timestamp"),
                pl.col("Open").first().alias("Target_Price"),
                pl.col("Close").last().alias("Market_Close"),
                pl.col("Volume").last().alias("Market_Close_Volume"),
                pl.col("Timestamp").last().alias("Market_Close_Timestamp"),
            ]
        )
        .sort(["Ticker", "Target_Timestamp"])
        .with_columns(
            [
                pl.col("Market_Close").shift(1).over("Ticker").alias("Reference_Price"),
                pl.col("Market_Close_Volume").shift(1).over("Ticker").alias("Reference_Volume"),
                pl.col("Market_Close_Timestamp").shift(1).over("Ticker").alias(
                    "Reference_Timestamp"
                ),
            ]
        )
        .drop_nulls(subset=["Reference_Price", "Reference_Timestamp"])
        .select(
            [
                "Ticker",
                "Target_Timestamp",
                "Target_Price",
                "Reference_Price",
                "Reference_Volume",
                "Reference_Timestamp",
            ]
        )
    )

    off_market_rows = with_market_time.filter(~pl.col("Is_Market_Hour")).sort(
        ["Ticker", "Timestamp"]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        assigned = (
            off_market_rows.join_asof(
                daily_market.sort(["Ticker", "Target_Timestamp"]),
                left_on="Timestamp",
                right_on="Target_Timestamp",
                by="Ticker",
                strategy="forward",
            )
            .drop_nulls(subset=["Target_Timestamp", "Target_Price", "Reference_Price"])
            .filter(pl.col("Timestamp") > pl.col("Reference_Timestamp"))
        )

    overnight = (
        assigned.group_by(["Ticker", "Target_Timestamp"], maintain_order=True)
        .agg(
            [
                pl.col("Timestamp").min().alias("Signal_Start"),
                pl.col("Timestamp").max().alias("Signal_End"),
                pl.col("Reference_Timestamp").first(),
                pl.col("Reference_Price").first(),
                pl.col("Reference_Volume").first().alias("Volume"),
                pl.col("Target_Price").first(),
                (
                    (pl.col("Sentiment_Mean") * pl.col("Post_Count")).sum()
                    / pl.col("Post_Count").sum()
                ).alias("Sentiment_Mean"),
                pl.col("Sentiment_Median").median().alias("Sentiment_Median"),
                pl.col("Sentiment_Mean").std().fill_null(0).alias("Sentiment_Std"),
                pl.col("Sentiment_Min").min().alias("Sentiment_Min"),
                pl.col("Sentiment_Max").max().alias("Sentiment_Max"),
                pl.col("Positive_Post_Count").sum().alias("Positive_Post_Count"),
                pl.col("Negative_Post_Count").sum().alias("Negative_Post_Count"),
                pl.col("Neutral_Post_Count").sum().alias("Neutral_Post_Count"),
                pl.col("Post_Count").sum().alias("Post_Count"),
            ]
        )
        .with_columns(
            [
                pl.lit("overnight").alias("Target_Type"),
                pl.col("Signal_End").alias("Timestamp"),
                pl.col("Reference_Price").alias("Open"),
                pl.col("Reference_Price").alias("High"),
                pl.col("Reference_Price").alias("Low"),
                pl.col("Reference_Price").alias("Close"),
                (
                    (
                        pl.col("Positive_Post_Count").cast(pl.Float64)
                        - pl.col("Negative_Post_Count").cast(pl.Float64)
                    )
                    / pl.col("Post_Count").cast(pl.Float64)
                ).alias("Bullishness_Index"),
                (
                    (pl.col("Target_Timestamp") - pl.col("Signal_End"))
                    .dt.total_seconds()
                    .cast(pl.Float64)
                    / 3600.0
                ).alias("Target_Horizon_Hours"),
                (
                    (pl.col("Target_Price") - pl.col("Reference_Price"))
                    / pl.col("Reference_Price")
                ).alias("Target_Return"),
            ]
        )
        .filter(pl.col("Target_Horizon_Hours") <= max_horizon_hours)
    )

    overnight = add_thresholded_direction_target(
        overnight, threshold=threshold, drop_neutral=drop_neutral
    )
    return overnight.select(_target_columns())


def build_hybrid_target_dataset(
    hourly_df: pl.DataFrame,
    threshold: float = TARGET_RETURN_THRESHOLD,
    max_intraday_horizon_hours: int = MAX_INTRADAY_HORIZON_HOURS,
    max_overnight_horizon_hours: int = MAX_OVERNIGHT_HORIZON_HOURS,
    drop_neutral: bool = True,
) -> pl.DataFrame:
    """Build the final Step 2 target dataset with intraday and overnight rows."""
    hourly_df = recompute_bullishness_index(hourly_df)
    intraday = build_intraday_target_rows(
        hourly_df,
        threshold=threshold,
        max_horizon_hours=max_intraday_horizon_hours,
        drop_neutral=drop_neutral,
    )
    overnight = build_overnight_target_rows(
        hourly_df,
        threshold=threshold,
        max_horizon_hours=max_overnight_horizon_hours,
        drop_neutral=drop_neutral,
    )
    return pl.concat([intraday, overnight], how="vertical").sort(["Ticker", "Timestamp"])
