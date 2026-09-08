"""Rolling-origin backtest of the latest candidate.

The output of this module is what the CI promotion gate reads. It writes a
machine-readable verdict so the workflow can fail without parsing logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.common import HEADS, get_logger, load_config, registered_name
from src.common.metrics import evaluate_all
from src.training import baselines
from src.training.dataset import (
    TARGET,
    feature_names,
    load_features,
    prepare_categoricals,
    rolling_origin_folds,
)

log = get_logger(__name__)


def backtest(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Refit once per fold.

    Slower than scoring one fixed model, and the only honest option: a model trained
    on the full history has already seen every test window in the backtest.
    """
    params = dict(cfg.model.params)
    params["alpha"] = 0.5
    num_round = params.pop("num_boost_round", 1000)
    params.pop("early_stopping_round", None)
    names = feature_names(df)
    # Encode categoricals once on the full frame: doing it per fold gives the same
    # store id a different code in train and test, which the booster reads as a
    # different store.
    encoded = prepare_categoricals(df)
    rows = []

    for fold, train, test in rolling_origin_folds(df, cfg):
        dtrain = lgb.Dataset(encoded.loc[train.index, names], label=train[TARGET])
        booster = lgb.train(params, dtrain, num_boost_round=num_round)
        y_pred = np.clip(booster.predict(encoded.loc[test.index, names]), 0, None)

        fold_metrics = {"fold": fold, **evaluate_all(test[TARGET], y_pred)}

        # Forecast the baseline across the whole fold, not just the first `horizon`
        # days of it: the train/test gap means a horizon-length baseline overlaps the
        # test window by a single day, and the gate would then compare against noise.
        span = int((test["date"].max() - train["date"].max()).days)
        base = baselines.seasonal_naive(train, horizon=max(span, int(cfg.data.horizon)))
        merged = test.merge(base, on=["store_id", "item_id", "date"], how="inner")
        if not merged.empty:
            fold_metrics["baseline_wmape"] = evaluate_all(merged[TARGET], merged["y_pred"])["wmape"]

        log.info("%s | wmape=%.4f", fold, fold_metrics["wmape"])
        rows.append(fold_metrics)

    return pd.DataFrame(rows)


def production_metric(cfg) -> float | None:
    """WMAPE of whatever is currently serving, or None on a cold registry.

    Read off the median head: it is the one the headline metric was computed from,
    and both heads are always promoted from the same run.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    name = registered_name(cfg.mlflow.registered_model, HEADS["p50"])
    try:
        versions = client.get_latest_versions(name, stages=["Production"])
    except Exception as exc:  # registry is empty on the very first run
        log.warning("no production model found for %s: %s", name, exc)
        return None
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get(cfg.promotion.primary_metric)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    df = load_features(Path(cfg.paths.features) / "sales_features")
    results = backtest(df, cfg)

    candidate = float(results["wmape"].mean())
    baseline = float(results["baseline_wmape"].mean()) if "baseline_wmape" in results else None
    incumbent = production_metric(cfg)

    reasons, passed = [], True
    if cfg.promotion.must_beat_baseline and baseline is not None and candidate >= baseline:
        passed = False
        reasons.append(f"candidate wmape {candidate:.4f} does not beat baseline {baseline:.4f}")
    if incumbent is not None:
        required = incumbent * (1 - float(cfg.promotion.min_improvement_pct) / 100)
        if candidate > required:
            passed = False
            reasons.append(
                f"candidate wmape {candidate:.4f} does not clear production {incumbent:.4f} "
                f"by {cfg.promotion.min_improvement_pct}% (needs <= {required:.4f})"
            )

    verdict = {
        "passed": passed,
        "candidate_wmape": candidate,
        "baseline_wmape": baseline,
        "production_wmape": incumbent,
        "reasons": reasons,
        "folds": results.to_dict(orient="records"),
    }

    reports = Path(cfg.paths.reports)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "backtest.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    results.to_csv(reports / "backtest_folds.csv", index=False)

    log.info("promotion gate: %s", "PASS" if passed else "FAIL")
    for reason in reasons:
        log.warning(reason)

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
