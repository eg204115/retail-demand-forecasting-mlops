"""Feature transforms.

Every function is a pure DataFrame -> DataFrame step so each one is unit-testable
against a handful of rows without standing up the whole pipeline.

Leakage rule enforced throughout: a feature for date d may only read data from
dates <= d - horizon. The lag floor is applied once, in `add_lag_features`, and
every rolling window is computed on top of already-lagged values.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

KEY = ["store_id", "item_id"]


def _series_window(order_col: str = "date") -> Window:
    return Window.partitionBy(*KEY).orderBy(F.col(order_col))


def add_calendar_features(df: DataFrame) -> DataFrame:
    """Deterministic date features - known for future dates, so safe at any horizon."""
    return (
        df.withColumn("dow", F.dayofweek("date"))
        .withColumn("day_of_month", F.dayofmonth("date"))
        .withColumn("week_of_year", F.weekofyear("date"))
        .withColumn("month", F.month("date"))
        .withColumn("year", F.year("date"))
        .withColumn("is_weekend", (F.dayofweek("date").isin([1, 7])).cast("int"))
        # Cyclical encoding so the tree doesn't have to learn that Sunday is next to Monday.
        .withColumn("dow_sin", F.sin(2 * F.lit(3.141592653589793) * F.col("dow") / 7))
        .withColumn("dow_cos", F.cos(2 * F.lit(3.141592653589793) * F.col("dow") / 7))
    )


def add_lag_features(df: DataFrame, lags: list[int], horizon: int) -> DataFrame:
    """Lagged demand, floored at the forecast horizon.

    At prediction time for date d we only know actuals up to d - horizon, so lag_1
    really means "the last value we would have had". Skipping this floor is the most
    common way these pipelines silently leak.
    """
    w = _series_window()
    for lag in lags:
        df = df.withColumn(f"lag_{lag}", F.lag("units", lag + horizon - 1).over(w))
    return df


def add_rolling_features(df: DataFrame, windows: list[int], horizon: int) -> DataFrame:
    """Rolling stats over the horizon-safe lagged series."""
    base = F.lag("units", horizon).over(_series_window())
    df = df.withColumn("_safe_units", base)

    for size in windows:
        w = _series_window().rowsBetween(-(size - 1), 0)
        df = (
            df.withColumn(f"roll_mean_{size}", F.avg("_safe_units").over(w))
            .withColumn(f"roll_std_{size}", F.stddev("_safe_units").over(w))
            .withColumn(f"roll_max_{size}", F.max("_safe_units").over(w))
            # Zero share captures intermittency, which drives the safety-stock quantile.
            .withColumn(
                f"zero_share_{size}",
                F.avg(F.when(F.col("_safe_units") == 0, 1.0).otherwise(0.0)).over(w),
            )
        )
    return df.drop("_safe_units")


def add_price_features(df: DataFrame, prices: DataFrame) -> DataFrame:
    """Price level, change and discount vs the series' own trailing average."""
    joined = df.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    w = _series_window()
    w52 = _series_window().rowsBetween(-364, -1)
    return (
        joined.withColumn("price_lag_1", F.lag("sell_price", 1).over(w))
        .withColumn(
            "price_change",
            (F.col("sell_price") - F.col("price_lag_1")) / F.col("price_lag_1"),
        )
        .withColumn("price_mean_52w", F.avg("sell_price").over(w52))
        .withColumn("price_ratio", F.col("sell_price") / F.col("price_mean_52w"))
        .withColumn("is_discounted", (F.col("price_ratio") < 0.95).cast("int"))
        .drop("price_lag_1")
    )


def add_event_features(df: DataFrame, calendar: DataFrame) -> DataFrame:
    """Holiday / SNAP flags. calendar is small - the caller should broadcast it."""
    cols = [c for c in ("date", "event_name_1", "event_type_1", "snap_CA") if c in calendar.columns]
    slim = calendar.select(*cols).withColumn("date", F.to_date("date"))
    out = df.join(F.broadcast(slim), on="date", how="left")
    if "event_name_1" in out.columns:
        out = out.withColumn("is_event", F.col("event_name_1").isNotNull().cast("int"))
    return out


def salted_key(col_name: str, skewed_keys: list[str], buckets: int) -> Column:
    """Spread a hot key across `buckets` shuffle partitions.

    Two of the ten M5 stores hold a disproportionate share of rows, so the shuffle
    for any per-store join lands on two tasks while the rest idle. Salting only the
    known-hot keys avoids inflating the small side of the join for everyone else.
    """
    is_hot = F.col(col_name).isin(skewed_keys)
    salt = (F.rand() * F.lit(buckets)).cast("int")
    return F.when(is_hot, F.concat_ws("#", F.col(col_name), salt)).otherwise(F.col(col_name))


def drop_leading_nulls(df: DataFrame, min_history_days: int) -> DataFrame:
    """Trim the warm-up rows where the long rolling windows are still undefined."""
    w = Window.partitionBy(*KEY).orderBy("date")
    return (
        df.withColumn("_row", F.row_number().over(w))
        .filter(F.col("_row") > min_history_days)
        .drop("_row")
    )
