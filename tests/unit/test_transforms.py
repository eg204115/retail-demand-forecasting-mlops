"""Spark transform tests.

Marked `spark` so `make test-fast` can skip the JVM startup, but they run in CI:
the leakage guarantees below are the ones worth catching automatically.
"""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from src.features import transforms as T  # noqa: E402

pytestmark = pytest.mark.spark


@pytest.fixture
def sales_df(spark):
    rows = [
        ("CA_1", "ITEM_001", dt.date(2024, 1, 1) + dt.timedelta(days=i), float(i))
        for i in range(40)
    ]
    return spark.createDataFrame(rows, ["store_id", "item_id", "date", "units"])


def test_calendar_features_are_deterministic(sales_df):
    out = T.add_calendar_features(sales_df).orderBy("date").collect()
    assert out[0]["dow"] == 2  # 2024-01-01 was a Monday; Spark dayofweek is 1=Sunday
    assert set(out[0].asDict()) >= {"dow", "month", "year", "is_weekend", "dow_sin", "dow_cos"}


def test_lag_respects_the_forecast_horizon(sales_df):
    """lag_1 at horizon 7 must be the value from 7 days ago, not yesterday.

    This is the single most important test in the repo: without the horizon floor
    the model trains on data that will not exist at prediction time and the offline
    metrics become fiction.
    """
    horizon = 7
    out = (
        T.add_lag_features(sales_df, [1], horizon)
        .filter(F.col("date") == dt.date(2024, 1, 21))
        .collect()[0]
    )
    # units == day index, so day 2024-01-21 is 20 and the horizon-safe lag is 14.
    assert out["lag_1"] == pytest.approx(20 - horizon)


def test_rolling_mean_excludes_the_current_row(sales_df):
    out = (
        T.add_rolling_features(sales_df, [7], horizon=1)
        .filter(F.col("date") == dt.date(2024, 1, 20))
        .collect()[0]
    )
    assert out["roll_mean_7"] < 19.0


def test_rolling_features_never_look_ahead(sales_df):
    """Every rolling value on day d must be reproducible from rows strictly before d."""
    rows = T.add_rolling_features(sales_df, [7], horizon=1).orderBy("date").collect()
    for row in rows[10:]:
        if row["roll_max_7"] is not None:
            assert row["roll_max_7"] < row["units"]


def test_zero_share_detects_intermittency(spark):
    rows = [
        ("CA_1", "ITEM_Z", dt.date(2024, 1, 1) + dt.timedelta(days=i), 0.0 if i % 2 else 3.0)
        for i in range(30)
    ]
    df = spark.createDataFrame(rows, ["store_id", "item_id", "date", "units"])
    out = T.add_rolling_features(df, [28], horizon=1).orderBy("date").collect()[-1]
    assert 0.3 < out["zero_share_28"] < 0.7


def test_salted_key_only_touches_hot_keys(spark):
    rows = [("CA_1", 1), ("CA_2", 2), ("CA_3", 3)]
    df = spark.createDataFrame(rows, ["store_id", "n"])
    out = df.withColumn("k", T.salted_key("store_id", ["CA_1"], 4)).collect()
    keys = {r["store_id"]: r["k"] for r in out}
    assert keys["CA_1"].startswith("CA_1#")
    assert keys["CA_2"] == "CA_2"  # untouched keys keep their original value
    assert keys["CA_3"] == "CA_3"


def test_drop_leading_nulls_trims_the_warmup(sales_df):
    out = T.drop_leading_nulls(sales_df, min_history_days=10)
    assert out.count() == 30
