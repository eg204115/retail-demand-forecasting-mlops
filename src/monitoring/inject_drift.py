"""Deliberately break the serving data so the monitoring can be seen catching it.

A monitoring stack nobody has watched fire is a config file, not a control. This
simulates a chain-wide promotion: prices drop, volume jumps, the price/demand
relationship the model learned no longer holds.

Writes a copy - the real feature table is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, load_config

log = get_logger(__name__)


def inject(df: pd.DataFrame, window_days: int, intensity: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    cutoff = out["date"].max() - pd.Timedelta(days=window_days - 1)
    mask = out["date"] >= cutoff
    log.info("perturbing %d of %d rows", int(mask.sum()), len(out))

    if "sell_price" in out.columns:
        out.loc[mask, "sell_price"] *= 1 - intensity
    if "price_ratio" in out.columns:
        out.loc[mask, "price_ratio"] *= 1 - intensity
    if "is_discounted" in out.columns:
        out.loc[mask, "is_discounted"] = 1

    # Promo lift with noise, so the shift is distributional rather than a constant offset.
    lift = 1 + intensity * 2 + rng.normal(0, 0.2, int(mask.sum()))
    for col in ("units", "lag_1", "lag_7", "roll_mean_7"):
        if col in out.columns:
            out.loc[mask, col] = out.loc[mask, col] * lift

    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--intensity", type=float, default=0.3, help="0.3 = a 30% price cut")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    source = Path(cfg.paths.features) / "sales_features"
    df = pd.read_parquet(source)
    df["date"] = pd.to_datetime(df["date"])

    drifted = inject(df, args.window_days, args.intensity)
    target = Path(cfg.paths.features) / "sales_features_drifted"
    drifted.to_parquet(target, index=False)

    log.info("drifted copy written to %s", target)
    log.info("now run: python -m src.monitoring.drift --input %s", target)


if __name__ == "__main__":
    main()
