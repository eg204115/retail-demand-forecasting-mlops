"""Nightly batch scoring: forecast every store x SKU and write the order plan.

Batch is the path that actually drives replenishment; the online API exists for
what-if queries and for the store app. Both load the same registered models through
`predictor.load_head`, so the two paths cannot disagree about what the model says.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.common import HEADS, get_logger, head_uri, load_config
from src.serving.predictor import load_head
from src.training.dataset import apply_category_levels, feature_names, load_features
from src.training.inventory import replenishment_plan

log = get_logger(__name__)


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The most recent feature row per series - the forecast origin."""
    return df.sort_values("date").groupby(["store_id", "item_id"], as_index=False).tail(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--stage", default="Production")
    parser.add_argument("--on-hand", default=None, help="CSV of store_id,item_id,on_hand")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    df = load_features(Path(cfg.paths.features) / "sales_features")
    origin = latest_rows(df)
    names = feature_names(df)

    heads = {}
    for label, head in HEADS.items():
        # Each quantile is its own registered model; loading one URI twice would make
        # the safety stock identically zero.
        uri = head_uri(cfg.mlflow.registered_model, head, args.stage)
        model, metadata = load_head(uri)
        frame = apply_category_levels(origin, metadata.get("categoricals") or {})[names]
        heads[label] = np.clip(model.predict(frame), 0, None)
        log.info("scored %s with %s", label, uri)

    forecasts = origin[["store_id", "item_id"]].copy()
    forecasts["p50"], forecasts["p90"] = heads["p50"], heads["p90"]
    forecasts["forecast_date"] = date.today()

    if args.on_hand:
        on_hand = pd.read_csv(args.on_hand)
    else:
        # No inventory feed wired up yet - assume empty shelves so the plan is the
        # full order-up-to level rather than a silently wrong number.
        on_hand = forecasts[["store_id", "item_id"]].assign(on_hand=0)

    plan = replenishment_plan(forecasts, on_hand, cfg)

    out_dir = Path(cfg.paths.artifacts) / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"replenishment_{date.today():%Y%m%d}.parquet"
    plan.to_parquet(out, index=False)

    log.info(
        "plan written to %s | %d SKUs | %d units ordered",
        out,
        len(plan),
        int(plan["order_qty"].sum()),
    )


if __name__ == "__main__":
    main()
