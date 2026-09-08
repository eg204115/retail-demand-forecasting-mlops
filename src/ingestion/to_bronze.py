"""Bronze layer: wide CSV -> long, partitioned, typed Parquet.

M5 ships one row per series with 1913 day columns. Everything downstream wants
(id, date, units), so the unpivot happens once here rather than in every job.
Partitioning by store keeps the later per-store window functions node-local.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common import get_logger, load_config
from src.features.spark_session import build_spark

log = get_logger(__name__)


def unpivot_sales(sales: DataFrame) -> DataFrame:
    """Wide d_1..d_N columns -> one row per (series, day).

    stack() keeps the reshape inside a single narrow transformation; a Python loop
    of unionAll() over 1913 columns builds a 1913-deep plan and never finishes.
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    id_cols = [c for c in id_cols if c in sales.columns]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    stack_expr = ", ".join(f"'{c}', `{c}`" for c in day_cols)
    return sales.select(
        *id_cols,
        F.expr(f"stack({len(day_cols)}, {stack_expr}) as (d, units)"),
    ).withColumn("units", F.col("units").cast("double"))


def build_bronze(spark: SparkSession, cfg) -> None:
    raw, bronze = Path(cfg.paths.raw), Path(cfg.paths.bronze)

    sales = spark.read.csv(str(raw / "sales_train_evaluation.csv"), header=True, inferSchema=True)
    calendar = spark.read.csv(str(raw / "calendar.csv"), header=True, inferSchema=True)
    prices = spark.read.csv(str(raw / "sell_prices.csv"), header=True, inferSchema=True)

    long_sales = unpivot_sales(sales)

    # calendar is tiny (~2k rows); broadcasting it avoids a full shuffle of the fact table.
    joined = (
        long_sales.join(F.broadcast(calendar.select("d", "date", "wm_yr_wk")), on="d", how="inner")
        .withColumn("date", F.to_date("date"))
        .drop("d")
    )

    (
        joined.repartition("store_id")
        .write.mode("overwrite")
        .partitionBy("store_id")
        .parquet(str(bronze / "sales"))
    )
    calendar.write.mode("overwrite").parquet(str(bronze / "calendar"))
    prices.write.mode("overwrite").parquet(str(bronze / "prices"))
    log.info("bronze written to %s", bronze)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    spark = build_spark(cfg)
    try:
        build_bronze(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
