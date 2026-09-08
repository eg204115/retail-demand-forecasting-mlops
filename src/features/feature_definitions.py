"""Feast feature definitions.

The point of registering these is the point-in-time join: `get_historical_features`
rewinds each feature to what was known at the label's timestamp, which is what makes
the training set honest. The same definitions serve the online store at request time,
so training and serving compute features from one source of truth.

Apply with:  cd conf && feast apply
"""

from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32

series = Entity(
    name="series",
    join_keys=["series_id"],
    value_type=ValueType.STRING,
    description="A store x SKU time series, e.g. CA_1_ITEM_042",
)

source = FileSource(
    name="sales_features_source",
    path="../data/features/sales_features",
    timestamp_field="event_timestamp",
)

demand_features = FeatureView(
    name="demand_features",
    entities=[series],
    # Serving may look back a week if a daily materialisation job is late.
    ttl=timedelta(days=7),
    schema=[
        Field(name="lag_1", dtype=Float32),
        Field(name="lag_7", dtype=Float32),
        Field(name="lag_14", dtype=Float32),
        Field(name="lag_28", dtype=Float32),
        Field(name="roll_mean_7", dtype=Float32),
        Field(name="roll_mean_28", dtype=Float32),
        Field(name="roll_mean_91", dtype=Float32),
        Field(name="roll_std_7", dtype=Float32),
        Field(name="roll_std_28", dtype=Float32),
        Field(name="zero_share_28", dtype=Float32),
        Field(name="sell_price", dtype=Float32),
        Field(name="price_ratio", dtype=Float32),
        Field(name="is_discounted", dtype=Int32),
        Field(name="is_event", dtype=Int32),
        Field(name="dow", dtype=Int32),
        Field(name="is_weekend", dtype=Int32),
    ],
    source=source,
    online=True,
    tags={"team": "octave", "domain": "retail-demand"},
)
