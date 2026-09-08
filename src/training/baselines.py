"""Baselines.

A forecasting project without a seasonal-naive comparison is unfalsifiable. These
are cheap, they run on every evaluation, and the promotion gate refuses to ship a
learned model that cannot beat them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["store_id", "item_id"]


def seasonal_naive(history: pd.DataFrame, horizon: int, season: int = 7) -> pd.DataFrame:
    """Predict each future day as the same weekday from the most recent full week."""
    out = []
    for keys, grp in history.groupby(KEY, sort=False):
        grp = grp.sort_values("date")
        tail = grp["units"].to_numpy()[-season:]
        if len(tail) < season:
            tail = np.pad(tail, (season - len(tail), 0), mode="edge")
        preds = np.resize(tail, horizon)
        last_date = grp["date"].max()
        out.append(
            pd.DataFrame(
                {
                    "store_id": keys[0],
                    "item_id": keys[1],
                    "date": pd.date_range(last_date, periods=horizon + 1, freq="D")[1:],
                    "y_pred": preds,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def moving_average(history: pd.DataFrame, horizon: int, window: int = 28) -> pd.DataFrame:
    """Flat forecast at the trailing mean - the floor any model must clear."""
    out = []
    for keys, grp in history.groupby(KEY, sort=False):
        mean = grp.sort_values("date")["units"].tail(window).mean()
        last_date = grp["date"].max()
        out.append(
            pd.DataFrame(
                {
                    "store_id": keys[0],
                    "item_id": keys[1],
                    "date": pd.date_range(last_date, periods=horizon + 1, freq="D")[1:],
                    "y_pred": float(mean),
                }
            )
        )
    return pd.concat(out, ignore_index=True)
