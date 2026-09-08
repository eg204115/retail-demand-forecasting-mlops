"""Model registry transitions.

Promotion is deliberately a separate step from training: a run being logged is not
the same event as a model being trusted in production, and CI needs somewhere to
put the approval.

Both quantile heads are registered and promoted together. Promoting them separately
would allow a P50 from one run to serve alongside a P90 from another, and the gap
between the two is exactly what sizes the safety stock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.common import HEADS, artifact_path, get_logger, load_config, registered_name

log = get_logger(__name__)


def latest_run_id(cfg) -> str:
    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment)
    if experiment is None:
        raise RuntimeError(f"experiment not found: {cfg.mlflow.experiment}")
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    if not runs:
        raise RuntimeError("no runs to promote")
    return runs[0].info.run_id


def register(cfg, run_id: str) -> dict[str, int]:
    """Register every head from one run. Returns {head: version}."""
    versions = {}
    for head in HEADS.values():
        name = registered_name(cfg.mlflow.registered_model, head)
        version = mlflow.register_model(f"runs:/{run_id}/{artifact_path(head)}", name)
        versions[head] = int(version.version)
        log.info("registered %s version %s", name, version.version)
    return versions


def promote(cfg, versions: dict[str, int], stage: str = "Production") -> None:
    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    for head, version in versions.items():
        name = registered_name(cfg.mlflow.registered_model, head)
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        log.info("%s version %s -> %s", name, version, stage)


def gate_passed(cfg) -> tuple[bool, list[str]]:
    """Read the backtest verdict. A missing verdict is not a pass."""
    verdict_path = Path(cfg.paths.reports) / "backtest.json"
    if not verdict_path.exists():
        return False, [f"no backtest verdict at {verdict_path}"]
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    return bool(verdict.get("passed")), list(verdict.get("reasons", []))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["register", "promote"])
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage", default="Production")
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="promote without a passing backtest verdict (break-glass)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    run_id = args.run_id or latest_run_id(cfg)

    if args.action == "register":
        register(cfg, run_id)
        return

    # Never promote a model the backtest gate has not cleared. A missing verdict
    # fails closed: "we did not check" is not "it passed".
    passed, reasons = gate_passed(cfg)
    if not passed and not args.skip_gate:
        raise SystemExit(f"refusing to promote, gate not cleared: {reasons}")
    if not passed:
        log.warning("--skip-gate set, promoting despite: %s", reasons)

    promote(cfg, register(cfg, run_id), args.stage)


if __name__ == "__main__":
    main()
