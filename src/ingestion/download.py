"""Fetch the raw M5 dataset into data/raw.

The Kaggle CLI needs ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY).
`--sample` writes a small synthetic dataset instead, so CI and a fresh clone can run
the whole pipeline without credentials.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, load_config

log = get_logger(__name__)

COMPETITION = "m5-forecasting-accuracy"
EXPECTED_FILES = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]


def download_kaggle(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s into %s", COMPETITION, raw_dir)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw_dir)],
        check=True,
    )
    archive = raw_dir / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()
    log.info("extracted: %s", sorted(p.name for p in raw_dir.glob("*.csv")))


def write_sample(raw_dir: Path, n_items: int = 40, n_stores: int = 3, n_days: int = 400) -> None:
    """Synthetic stand-in with the same schema and the same seasonality shape."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")

    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "d": [f"d_{i + 1}" for i in range(n_days)],
            "wm_yr_wk": (np.arange(n_days) // 7) + 11101,
            "weekday": dates.day_name(),
            "wday": dates.dayofweek + 1,
            "month": dates.month,
            "year": dates.year,
            "snap_CA": rng.integers(0, 2, n_days),
        }
    )
    calendar["event_name_1"] = np.where(rng.random(n_days) < 0.05, "Event", None)

    items = [f"ITEM_{i:03d}" for i in range(n_items)]
    stores = [f"CA_{i + 1}" for i in range(n_stores)]
    rows, prices = [], []
    for store in stores:
        for item in items:
            base = rng.gamma(2.0, 2.0)
            weekly = 1 + 0.35 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
            trend = np.linspace(1.0, rng.uniform(0.8, 1.3), n_days)
            sales = rng.poisson(np.clip(base * weekly * trend, 0.05, None))
            row = {"id": f"{item}_{store}", "item_id": item, "store_id": store}
            row.update({f"d_{i + 1}": int(v) for i, v in enumerate(sales)})
            rows.append(row)
            for wk in calendar["wm_yr_wk"].unique():
                prices.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": wk,
                        "sell_price": round(float(rng.uniform(1.0, 12.0)), 2),
                    }
                )

    pd.DataFrame(rows).to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(raw_dir / "calendar.csv", index=False)
    pd.DataFrame(prices).to_csv(raw_dir / "sell_prices.csv", index=False)
    log.info("wrote sample dataset: %d series x %d days", n_items * n_stores, n_days)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--sample", action="store_true", help="generate synthetic data instead")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(cfg.paths.raw)

    if args.sample:
        write_sample(raw_dir)
        return
    if all((raw_dir / f).exists() for f in EXPECTED_FILES):
        log.info("raw files already present, skipping download")
        return
    download_kaggle(raw_dir)


if __name__ == "__main__":
    main()
