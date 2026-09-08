"""Train the quantile demand model and log everything to MLflow.

Two models are fitted, one per quantile: P50 is the point forecast, P90 sizes the
safety stock. A single conditional-mean model cannot do the second job, and the
inventory decision is the whole reason this project exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.common import artifact_path, get_logger, head_for, load_config
from src.common.metrics import coverage, evaluate_all, pinball_loss
from src.training import baselines
from src.training.dataset import (
    CATEGORICALS,
    TARGET,
    chronological_split,
    load_features,
    prepare_categoricals,
)

log = get_logger(__name__)


def _fit_one(
    quantile: float,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: list[str],
    params: dict,
) -> lgb.Booster:
    p = dict(params)
    p["alpha"] = quantile
    num_round = p.pop("num_boost_round", 1000)
    early_stop = p.pop("early_stopping_round", 100)

    dtrain = lgb.Dataset(train[feature_names], label=train[TARGET])
    dvalid = lgb.Dataset(valid[feature_names], label=valid[TARGET], reference=dtrain)

    log.info("fitting quantile %.2f on %d rows", quantile, len(train))
    return lgb.train(
        p,
        dtrain,
        num_boost_round=num_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )


def category_levels(df: pd.DataFrame) -> dict[str, list[str]]:
    """The exact category order LightGBM encoded, recorded for serving.

    A pandas `category` column is encoded by position. If the serving frame derives
    its categories from whatever ids happened to be in one request, the codes no
    longer mean the same thing and the booster scores a different feature. Shipping
    the levels with the model is the cheapest defence against that skew.
    """
    return {
        col: [str(v) for v in df[col].cat.categories]
        for col in CATEGORICALS
        if col in df.columns and str(df[col].dtype) == "category"
    }


def _baseline_metrics(df: pd.DataFrame, split, cfg) -> dict[str, float]:
    """Seasonal naive on the test window, aligned on (store, item, date)."""
    history = df[df["date"] < split.test["date"].min()]
    if history.empty or split.test.empty:
        return {}
    # Forecast far enough to cover the whole test window. Using `horizon` alone
    # overlaps only the first days of the window, and a baseline measured on a
    # handful of days is noise the promotion gate would then act on.
    span = int((split.test["date"].max() - history["date"].max()).days)
    preds = baselines.seasonal_naive(history, horizon=max(span, int(cfg.data.horizon)))
    merged = split.test.merge(preds, on=["store_id", "item_id", "date"], how="inner")
    if merged.empty:
        log.warning("no overlap between baseline forecast and test window")
        return {}
    return {f"baseline_{k}": v for k, v in evaluate_all(merged[TARGET], merged["y_pred"]).items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    df = load_features(Path(cfg.paths.features) / "sales_features")
    split = chronological_split(df, cfg)
    quantiles = [float(q) for q in cfg.model.quantiles]

    # Categoricals are encoded once, on the full frame, so train/valid/test share one
    # level ordering; encoding each split separately would give the same store id a
    # different code in each.
    encoded = prepare_categoricals(df)
    levels = category_levels(encoded)
    train = encoded.loc[split.train.index]
    valid = encoded.loc[split.valid.index]
    test = encoded.loc[split.test.index]

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "horizon": cfg.data.horizon,
                "lags": str(list(cfg.data.lags)),
                "rolling_windows": str(list(cfg.data.rolling_windows)),
                "quantiles": str(quantiles),
                "n_features": len(split.feature_names),
                "n_train_rows": len(train),
                **{f"lgb_{k}": v for k, v in dict(cfg.model.params).items()},
            }
        )
        mlflow.set_tags({"project": cfg.project, "stage": "training"})

        boosters, preds = {}, {}

        for q in quantiles:
            head = head_for(q)
            booster = _fit_one(q, train, valid, split.feature_names, dict(cfg.model.params))
            boosters[q] = booster
            preds[q] = np.clip(booster.predict(test[split.feature_names]), 0, None)
            mlflow.log_metric(f"pinball_{head}", pinball_loss(test[TARGET], preds[q], q))
            mlflow.log_metric(f"best_iteration_{head}", booster.best_iteration)
            # The category levels travel with the model, so the serving container
            # never has to guess how the training frame was encoded.
            mlflow.lightgbm.log_model(
                booster,
                artifact_path=artifact_path(head),
                metadata={
                    "quantile": q,
                    "head": head,
                    "feature_names": split.feature_names,
                    "categoricals": levels,
                },
            )

        # Headline metrics come from the median model.
        median_q = min(quantiles, key=lambda q: abs(q - 0.5))
        metrics = evaluate_all(test[TARGET], preds[median_q])
        upper_q = max(quantiles)
        if upper_q != median_q:
            metrics[f"coverage_{head_for(upper_q)}"] = coverage(test[TARGET], preds[upper_q])
        metrics.update(_baseline_metrics(df, split, cfg))
        mlflow.log_metrics(metrics)

        importance = pd.DataFrame(
            {
                "feature": boosters[median_q].feature_name(),
                "gain": boosters[median_q].feature_importance("gain"),
            }
        ).sort_values("gain", ascending=False)

        reports = Path(cfg.paths.reports)
        reports.mkdir(parents=True, exist_ok=True)
        importance.to_csv(reports / "feature_importance.csv", index=False)
        (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (reports / "feature_schema.json").write_text(
            json.dumps({"feature_names": split.feature_names, "categoricals": levels}, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(reports / "feature_importance.csv"))
        mlflow.log_artifact(str(reports / "metrics.json"))
        mlflow.log_artifact(str(reports / "feature_schema.json"))

        log.info("run %s | wmape=%.4f", run.info.run_id, metrics["wmape"])
        if "baseline_wmape" in metrics:
            lift = 1 - metrics["wmape"] / metrics["baseline_wmape"]
            log.info("lift over seasonal naive: %.1f%%", lift * 100)


if __name__ == "__main__":
    main()
