from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.config import Config


@pytest.fixture(scope="session")
def cfg() -> Config:
    from src.common import load_config

    return load_config("conf/config.yaml")


@pytest.fixture
def panel() -> pd.DataFrame:
    """Small deterministic sales panel: 4 series x 120 days."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rows = []
    for store in ("CA_1", "CA_2"):
        for item in ("ITEM_001", "ITEM_002"):
            units = rng.poisson(5, len(dates)).astype(float)
            rows.append(
                pd.DataFrame(
                    {
                        "store_id": store,
                        "item_id": item,
                        "date": dates,
                        "units": units,
                        "sell_price": rng.uniform(2, 6, len(dates)).round(2),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def spark():
    """Local single-node SparkSession, reused across the whole session.

    Session-scoped because JVM startup is several seconds and dominates the suite.
    """
    pyspark = pytest.importorskip("pyspark")
    session = (
        pyspark.sql.SparkSession.builder.appName("shelfcast-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
