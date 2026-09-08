"""Bronze -> silver -> feature table.

Writes a single point-in-time-correct feature table that both training and Feast
materialisation read, so the offline and online paths cannot drift apart.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common import get_logger, load_config
from src.features import transforms as T
from src.features.spark_session import build_spark

log = get_logger(__name__)


def build(spark: SparkSession, cfg) -> DataFrame:
    bronze, horizon = Path(cfg.paths.bronze), int(cfg.data.horizon)

    sales = spark.read.parquet(str(bronze / "sales"))
    calendar = spark.read.parquet(str(bronze / "calendar"))
    prices = spark.read.parquet(str(bronze / "prices"))

    df = T.add_calendar_features(sales)
    df = T.add_event_features(df, calendar)
    df = T.add_price_features(df, prices)

    # Lags and rolling stats share the same partitioning; repartitioning once here
    # means the ~20 window functions below run without a shuffle between them.
    df = df.repartition(
        int(cfg.spark.shuffle_partitions), "store_id", "item_id"
    ).sortWithinPartitions("store_id", "item_id", "date")

    df = T.add_lag_features(df, list(cfg.data.lags), horizon)
    df = T.add_rolling_features(df, list(cfg.data.rolling_windows), horizon)
    df = T.drop_leading_nulls(df, max(cfg.data.rolling_windows) + horizon)

    # Feast needs an event timestamp and a stable entity key.
    return df.withColumn("event_timestamp", F.col("date").cast("timestamp")).withColumn(
        "series_id", F.concat_ws("_", "store_id", "item_id")
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--explain", action="store_true", help="print the physical plan and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    spark = build_spark(cfg)
    started = time.perf_counter()
    try:
        features = build(spark, cfg)
        if args.explain:
            features.explain(mode="formatted")
            return

        out = Path(cfg.paths.features) / "sales_features"
        features.write.mode("overwrite").partitionBy("store_id").parquet(str(out))
        elapsed = time.perf_counter() - started
        log.info("feature table written to %s in %.1fs", out, elapsed)
        log.info("columns: %d", len(features.columns))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
