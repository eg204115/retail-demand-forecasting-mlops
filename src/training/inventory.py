"""Turn a forecast distribution into an order quantity.

This is the step that makes the model a decision rather than a number, and it is
where the P90 head earns its keep: safety stock is the gap between the service-level
quantile and the median, scaled over the protection interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def protection_interval(lead_time_days: int, review_period_days: int) -> int:
    """Demand at risk covers the lead time plus the next review cycle."""
    return int(lead_time_days) + int(review_period_days)


def order_up_to_level(
    forecast_median: np.ndarray,
    forecast_upper: np.ndarray,
    lead_time_days: int,
    review_period_days: int,
) -> np.ndarray:
    """Order-up-to level = expected demand over the interval + safety stock."""
    days = protection_interval(lead_time_days, review_period_days)
    median = np.asarray(forecast_median, float)
    upper = np.asarray(forecast_upper, float)
    expected = median * days
    # Daily quantile spread scaled by sqrt(days): errors across days are not perfectly
    # correlated, so multiplying the spread by `days` would badly over-stock.
    safety = np.maximum(upper - median, 0) * np.sqrt(days)
    return expected + safety


def replenishment_plan(forecasts: pd.DataFrame, on_hand: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per (store, SKU) with the quantity to order today."""
    inv = cfg.inventory
    days = protection_interval(inv.lead_time_days, inv.review_period_days)

    agg = forecasts.groupby(["store_id", "item_id"], as_index=False).agg(
        p50=("p50", "mean"), p90=("p90", "mean")
    )
    merged = agg.merge(on_hand, on=["store_id", "item_id"], how="left")
    merged["on_hand"] = merged.get("on_hand", pd.Series(0, index=merged.index)).fillna(0)

    merged["order_up_to"] = order_up_to_level(
        merged["p50"], merged["p90"], inv.lead_time_days, inv.review_period_days
    )
    merged["safety_stock"] = (merged["order_up_to"] - merged["p50"] * days).round(2)
    merged["order_qty"] = np.ceil(np.maximum(merged["order_up_to"] - merged["on_hand"], 0)).astype(
        int
    )
    return merged
