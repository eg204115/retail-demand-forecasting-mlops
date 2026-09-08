"""Data and prediction drift, gated for CI.

Two questions, deliberately separated:
  1. Have the inputs moved? (answerable immediately, no labels needed)
  2. Has accuracy dropped? (the thing you care about, but only after actuals land)

Input drift is the early warning; performance decay is the confirmation. Retraining
on input drift alone retrains on every holiday, so the alert routes to a human and
the automated trigger waits for the performance signal.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.common import get_logger, load_config
from src.common.metrics import wmape

log = get_logger(__name__)

TARGET = "units"

MONITORED = [
    "lag_1",
    "lag_7",
    "roll_mean_7",
    "roll_mean_28",
    "roll_std_28",
    "zero_share_28",
    "sell_price",
    "price_ratio",
    "is_discounted",
    "is_event",
]


def split_windows(df: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference = the window the model was fitted on; current = the latest window."""
    last = df["date"].max()
    current_start = last - timedelta(days=window_days - 1)
    reference_start = current_start - timedelta(days=window_days)
    current = df[df["date"] >= current_start]
    reference = df[(df["date"] >= reference_start) & (df["date"] < current_start)]
    return reference, current


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path) -> dict:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report

    columns = [c for c in MONITORED if c in reference.columns]
    # Evidently looks for columns literally named `target`/`prediction` unless told
    # otherwise; without this mapping the target-drift preset silently reports on
    # nothing, which looks identical to "no drift".
    mapping = ColumnMapping(
        target=TARGET if TARGET in reference.columns else None,
        prediction="prediction" if "prediction" in reference.columns else None,
    )
    metrics = [DataDriftPreset(columns=columns)]
    if mapping.target is not None:
        metrics.append(TargetDriftPreset())
    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    out_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_dir / "drift_report.html"))
    result = report.as_dict()
    (out_dir / "drift_report.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def summarise(result: dict) -> dict:
    """Pull the two numbers a gate needs out of the Evidently payload."""
    for metric in result.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            summary = metric["result"]
            return {
                "n_features": summary.get("number_of_columns"),
                "n_drifted": summary.get("number_of_drifted_columns"),
                "drift_share": summary.get("share_of_drifted_columns", 0.0),
                "drifted_features": [
                    name
                    for name, col in summary.get("drift_by_columns", {}).items()
                    if col.get("drift_detected")
                ],
            }
    return {"drift_share": 0.0, "drifted_features": []}


def performance_check(df: pd.DataFrame, cfg) -> dict:
    """Live WMAPE against the value recorded at training time, once actuals exist."""
    metrics_path = Path(cfg.paths.reports) / "metrics.json"
    if not metrics_path.exists() or "prediction" not in df.columns:
        return {}
    trained = json.loads(metrics_path.read_text(encoding="utf-8")).get("wmape")
    scored = df.dropna(subset=["prediction", "units"])
    if trained is None or scored.empty:
        return {}
    live = wmape(scored["units"], scored["prediction"])
    return {
        "live_wmape": live,
        "training_wmape": trained,
        "degradation": (live - trained) / trained if trained else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--input", default=None, help="override the feature table path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    path = args.input or (Path(cfg.paths.features) / "sales_features")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    reference, current = split_windows(df, int(cfg.monitoring.reference_window_days))
    log.info("reference=%d rows | current=%d rows", len(reference), len(current))

    out_dir = Path(cfg.paths.reports)
    if reference.empty or current.empty:
        raise SystemExit(
            f"not enough history to compare: reference={len(reference)} current={len(current)} rows"
        )

    summary = summarise(evidently_report(reference, current, out_dir))
    summary.update(performance_check(current, cfg))

    share_breached = summary["drift_share"] > float(cfg.monitoring.drift_share_threshold)
    degradation = summary.get("degradation")
    perf_breached = degradation is not None and degradation > float(
        cfg.monitoring.wmape_degradation_threshold
    )
    summary["retrain_recommended"] = bool(share_breached or perf_breached)

    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("drift share %.2f | drifted: %s", summary["drift_share"], summary["drifted_features"])
    if summary["retrain_recommended"]:
        log.warning("DRIFT GATE TRIPPED - retraining recommended")

    raise SystemExit(1 if summary["retrain_recommended"] else 0)


if __name__ == "__main__":
    main()
