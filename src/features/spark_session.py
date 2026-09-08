"""SparkSession factory.

Every tuning decision here is measured in docs/spark-optimisation.md. Keeping them
in one place means the local run, the CI run and the cluster run share a config and
a benchmark can attribute a change to a single knob.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from src.common import get_logger

log = get_logger(__name__)


def build_spark(cfg, app_name: str | None = None) -> SparkSession:
    s = cfg.spark
    broadcast_bytes = int(s.broadcast_threshold_mb) * 1024 * 1024

    builder = (
        SparkSession.builder.appName(app_name or s.app_name)
        .master(s.master)
        .config("spark.sql.shuffle.partitions", s.shuffle_partitions)
        .config("spark.driver.memory", s.driver_memory)
        .config("spark.executor.memory", s.executor_memory)
        # AQE coalesces the post-shuffle partitions that the static setting above
        # over-provisions, and converts sort-merge joins to broadcast at runtime.
        .config("spark.sql.adaptive.enabled", str(s.adaptive_enabled).lower())
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", str(s.skew_join_enabled).lower())
        .config("spark.sql.autoBroadcastJoinThreshold", broadcast_bytes)
        # Arrow makes the pandas UDFs in transforms.py roughly an order faster.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.files.maxPartitionBytes", "128m")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info(
        "spark %s | master=%s | shuffle.partitions=%s | broadcast=%sMB",
        spark.version,
        s.master,
        s.shuffle_partitions,
        s.broadcast_threshold_mb,
    )
    return spark
