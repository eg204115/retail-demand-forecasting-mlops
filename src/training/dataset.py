"""Turn the feature table into train/valid/test frames with an honest split.

The split is chronological with a `horizon` gap between train and validation. A
random split on time series is the second most common leak after unfloored lags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common import get_logger

log = get_logger(__name__)

TARGET = "units"
NON_FEATURES = {
    TARGET,
    "date",
    "event_timestamp",
    "id",
    "series_id",
    "wm_yr_wk",
    "event_name_1",
    "event_type_1",
}
CATEGORICALS = ["store_id", "item_id", "dept_id", "cat_id", "state_id"]


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_names: list[str]


def load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def chronological_split(df: pd.DataFrame, cfg) -> Split:
    horizon = int(cfg.data.horizon)
    valid_days, test_days = int(cfg.data.valid_days), int(cfg.data.test_days)

    last = df["date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    # Gap: training must not see anything a real forecast origin would not have.
    train_end = valid_start - pd.Timedelta(days=horizon)

    split = Split(
        train=df[df["date"] <= train_end],
        valid=df[(df["date"] > valid_start) & (df["date"] < test_start)],
        test=df[df["date"] >= test_start],
        feature_names=feature_names(df),
    )
    log.info(
        "split | train<=%s (%d) | valid (%d) | test>=%s (%d) | gap=%dd",
        train_end.date(),
        len(split.train),
        len(split.valid),
        test_start.date(),
        len(split.test),
        horizon,
    )
    return split


def rolling_origin_folds(df: pd.DataFrame, cfg):
    """Yield (train, test) frames for a rolling-origin backtest.

    One holdout tells you almost nothing about a seasonal series - a lucky month
    hides a broken model. Folds walk the origin forward so the reported metric is
    an average over several regimes.
    """
    folds, step = int(cfg.data.backtest_folds), int(cfg.data.fold_step)
    horizon = int(cfg.data.horizon)
    last = df["date"].max()

    for i in range(folds):
        test_end = last - pd.Timedelta(days=i * step)
        test_start = test_end - pd.Timedelta(days=step - 1)
        train_end = test_start - pd.Timedelta(days=horizon)
        train = df[df["date"] <= train_end]
        test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        if train.empty or test.empty:
            continue
        yield f"fold_{folds - i}", train, test


def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM handles pandas `category` dtype natively - no one-hot explosion."""
    out = df.copy()
    for col in CATEGORICALS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def apply_category_levels(df: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Re-encode categoricals with the exact levels the model was trained on.

    LightGBM stores category codes, not labels. Rebuilding categories from whatever
    values one scoring frame happens to contain gives the same store a different
    code than it had in training - the classic silent training/serving skew.
    """
    out = df.copy()
    for col, cats in levels.items():
        if col in out.columns:
            # astype(CategoricalDtype) maps a value the model never saw to NaN, which
            # LightGBM treats as missing. Constructing a Categorical directly would
            # do the same today and raise in pandas 4.
            dtype = pd.CategoricalDtype(categories=list(cats))
            out[col] = out[col].astype(str).astype(dtype)
    return out
