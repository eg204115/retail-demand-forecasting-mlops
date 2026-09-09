"""Training/serving skew guards for the online feature frame.

Every case below covers a failure that returns a 200 with a wrong number rather than
an error, which is the only kind worth a test at this layer.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.serving.predictor import align_to_model, build_frame
from src.training.dataset import apply_category_levels


class _Model:
    """Stands in for a Booster: alignment only ever asks for the column order."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def feature_name(self) -> list[str]:
        return self._names


def test_day_of_week_matches_the_spark_convention():
    """Spark's dayofweek is 1=Sunday, and serving must encode it the same way.

    An off-by-one here is invisible: every forecast still returns, the model is just
    consistently reading Monday's coefficient on Sunday.
    """
    # 2024-01-07 is a Sunday, 2024-01-08 a Monday.
    frame = build_frame("CA_1", "ITEM_1", origin=dt.date(2024, 1, 6), horizon=2)
    by_date = dict(zip(frame["target_date"], frame["dow"], strict=True))
    assert by_date[dt.date(2024, 1, 7)] == 1
    assert by_date[dt.date(2024, 1, 8)] == 2


def test_weekend_flag_covers_saturday_and_sunday():
    frame = build_frame("CA_1", "ITEM_1", origin=dt.date(2024, 1, 4), horizon=4)
    flags = dict(zip(frame["target_date"], frame["is_weekend"], strict=True))
    assert flags[dt.date(2024, 1, 5)] == 0  # Friday
    assert flags[dt.date(2024, 1, 6)] == 1  # Saturday
    assert flags[dt.date(2024, 1, 7)] == 1  # Sunday


def test_one_row_per_forecast_day():
    frame = build_frame("CA_1", "ITEM_1", origin=dt.date(2024, 1, 1), horizon=14)
    assert len(frame) == 14
    assert frame["target_date"].min() == dt.date(2024, 1, 2)


def test_alignment_reorders_to_the_training_column_order():
    frame = pd.DataFrame({"b": [1], "a": [2], "unused": [3]})
    aligned = align_to_model(frame, _Model(["a", "b"]))
    assert list(aligned.columns) == ["a", "b"]


def test_missing_training_columns_become_nan_not_dropped():
    aligned = align_to_model(pd.DataFrame({"a": [1]}), _Model(["a", "lag_1"]))
    assert list(aligned.columns) == ["a", "lag_1"]
    assert aligned["lag_1"].isna().all()


def test_category_codes_follow_the_training_levels():
    """The same store must encode to the same code it had in training.

    Rebuilding categories from one request's values is the classic silent skew: a
    frame containing only CA_3 would encode it as 0, which the booster reads as
    whichever store happened to be first at training time.
    """
    levels = {"store_id": ["CA_1", "CA_2", "CA_3"]}
    one_store = apply_category_levels(pd.DataFrame({"store_id": ["CA_3"]}), levels)
    assert list(one_store["store_id"].cat.categories) == levels["store_id"]
    assert one_store["store_id"].cat.codes.tolist() == [2]


def test_alignment_applies_training_levels_when_they_are_known():
    frame = pd.DataFrame({"store_id": ["CA_2"], "lag_1": [1.0]})
    aligned = align_to_model(
        frame, _Model(["store_id", "lag_1"]), {"store_id": ["CA_1", "CA_2", "CA_3"]}
    )
    assert aligned["store_id"].cat.codes.tolist() == [1]


def test_unknown_category_does_not_crash_scoring():
    """A store the model never saw becomes NaN, not a new code silently reused."""
    aligned = apply_category_levels(pd.DataFrame({"store_id": ["TX_9"]}), {"store_id": ["CA_1"]})
    assert aligned["store_id"].isna().all()
