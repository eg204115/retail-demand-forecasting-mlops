"""Data contracts.

These run before training in the DAG. Catching a schema break here costs a failed
task; catching it after a model has trained on nulls costs a week of bad orders.
"""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

sales_schema = DataFrameSchema(
    {
        "store_id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        # Negative units means a returns row leaked into the sales feed.
        "units": Column(float, Check.ge(0), nullable=False),
    },
    strict=False,
    unique=["store_id", "item_id", "date"],
    name="sales",
)

features_schema = DataFrameSchema(
    {
        "store_id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        "units": Column(float, Check.ge(0)),
        # Lags may be null only during the warm-up, which build_features trims.
        "lag_1": Column(float, Check.ge(0), nullable=True),
        "lag_7": Column(float, Check.ge(0), nullable=True),
        "roll_mean_28": Column(float, Check.ge(0), nullable=True),
        "zero_share_28": Column(float, Check.in_range(0, 1), nullable=True),
        "sell_price": Column(float, Check.gt(0), nullable=True),
        "is_weekend": Column(int, Check.isin([0, 1])),
    },
    strict=False,
    name="features",
)


def validate(df, schema: DataFrameSchema, lazy: bool = True):
    """Collect every failure at once rather than stopping at the first."""
    return schema.validate(df, lazy=lazy)


__all__ = ["sales_schema", "features_schema", "validate", "pa"]
