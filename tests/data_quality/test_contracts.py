from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pandera")

from pandera.errors import SchemaError, SchemaErrors  # noqa: E402

from tests.data_quality.schemas import sales_schema, validate  # noqa: E402

# `validate` collects failures lazily, so a broken frame raises SchemaErrors; the
# uniqueness check is reported eagerly as a single SchemaError.
SCHEMA_FAILURE = (SchemaError, SchemaErrors)


def _frame(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["A", "A"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "units": [1.0, 2.0],
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


def test_valid_frame_passes():
    assert len(validate(_frame(), sales_schema)) == 2


def test_negative_units_are_rejected():
    with pytest.raises(SCHEMA_FAILURE):
        validate(_frame(units=[-1.0, 2.0]), sales_schema)


def test_duplicate_series_days_are_rejected():
    dupe = _frame()
    dupe.loc[1, "date"] = dupe.loc[0, "date"]
    with pytest.raises(SCHEMA_FAILURE):
        validate(dupe, sales_schema)


def test_null_keys_are_rejected():
    with pytest.raises(SCHEMA_FAILURE):
        validate(_frame(store_id=[None, "CA_1"]), sales_schema)
