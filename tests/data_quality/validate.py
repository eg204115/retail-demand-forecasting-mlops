"""CLI wrapper so Airflow and CI can run the contracts as a pipeline step."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import get_logger, load_config
from tests.data_quality.schemas import features_schema, sales_schema, validate

log = get_logger(__name__)

LAYERS = {"bronze": ("sales", sales_schema), "features": ("sales_features", features_schema)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=sorted(LAYERS), default="features")
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--sample-rows", type=int, default=200_000)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    name, schema = LAYERS[args.layer]
    base = Path(cfg.paths.bronze if args.layer == "bronze" else cfg.paths.features)

    df = pd.read_parquet(base / name)
    if len(df) > args.sample_rows:
        # Validating 50M rows on every DAG run is not worth the wall clock; a large
        # random sample catches structural breaks just as reliably.
        df = df.sample(args.sample_rows, random_state=42)
    df["date"] = pd.to_datetime(df["date"])

    try:
        validate(df, schema)
    except Exception as exc:
        log.error("%s failed validation:\n%s", args.layer, exc)
        raise SystemExit(1) from exc

    log.info("%s passed validation on %d rows", args.layer, len(df))


if __name__ == "__main__":
    main()
