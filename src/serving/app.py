"""FastAPI scoring service.

Exposes the forecast, the replenishment decision built on top of it, and the
Prometheus metrics Grafana scrapes. /health is shallow (is the process up) and
/ready is deep (can we actually score), because a readiness probe that only checks
the process keeps routing traffic to a container with no model.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.common import get_logger, load_config
from src.serving import predictor
from src.serving.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    DayForecast,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)
from src.training.inventory import order_up_to_level

log = get_logger(__name__)
cfg = load_config(os.getenv("CONFIG_PATH", "conf/config.yaml"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model at startup, not on the first request, so p99 is not a lie."""
    bundle = predictor.load_models()
    log.info("startup complete | model_loaded=%s", bundle.loaded)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Shelfcast demand forecasting API",
    description="SKU-level demand forecasts and replenishment quantities.",
    version="0.1.0",
)

PREDICTIONS = Counter("shelfcast_predictions_total", "Forecast rows returned", ["endpoint"])
FEATURE_MISSES = Counter(
    "shelfcast_feature_misses_total", "Online feature lookups that returned nothing"
)
PREDICT_LATENCY = Histogram(
    "shelfcast_predict_seconds", "End-to-end prediction latency", ["endpoint"]
)
# Logged so drift monitoring can compare live inputs against the training reference.
PREDICTION_VALUE = Histogram(
    "shelfcast_prediction_units", "Predicted units per day", buckets=(0, 1, 2, 5, 10, 25, 50, 100)
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


def _forecast_one(req: ForecastRequest) -> ForecastResponse:
    bundle = predictor.load_models()
    if not bundle.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    origin = req.forecast_date or date.today()
    frame = predictor.build_frame(req.store_id, req.item_id, origin, req.horizon, req.features)
    # A frame with no lag columns means the online store returned nothing for this
    # series: the forecast is calendar-only and worth counting, not worth hiding.
    if not any(col in frame.columns for col in ("lag_1", "lag_7", "roll_mean_7")):
        FEATURE_MISSES.inc()

    preds = bundle.predict(frame)

    days = [
        DayForecast(
            date=frame["target_date"].iloc[i],
            p50=round(float(preds["p50"][i]), 3),
            p90=round(float(preds.get("p90", preds["p50"])[i]), 3),
        )
        for i in range(len(frame))
    ]
    for day in days:
        PREDICTION_VALUE.observe(day.p50)
    PREDICTIONS.labels(endpoint="forecast").inc(len(days))

    inv = cfg.inventory
    level = order_up_to_level(
        [sum(d.p50 for d in days) / len(days)],
        [sum(d.p90 for d in days) / len(days)],
        inv.lead_time_days,
        inv.review_period_days,
    )

    return ForecastResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        horizon=req.horizon,
        model_version=bundle.version,
        forecasts=days,
        order_qty=int(round(float(level[0]))),
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    bundle = predictor.load_models()
    return HealthResponse(
        status="ok",
        model_loaded=bundle.loaded,
        model_version=bundle.version if bundle.loaded else None,
        feature_store="up" if predictor.get_feature_store() is not None else "unavailable",
    )


@app.get("/ready", tags=["ops"])
def ready() -> dict[str, str]:
    """Deep check: score a throwaway row and make sure a number comes back."""
    bundle = predictor.load_models()
    if not bundle.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        _forecast_one(ForecastRequest(store_id="__probe__", item_id="__probe__", horizon=1))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"scoring path broken: {exc}") from exc
    return {"status": "ready"}


@app.post("/predict", response_model=ForecastResponse, tags=["forecast"])
def predict(req: ForecastRequest) -> ForecastResponse:
    started = time.perf_counter()
    try:
        return _forecast_one(req)
    finally:
        PREDICT_LATENCY.labels(endpoint="predict").observe(time.perf_counter() - started)


@app.post("/predict/batch", response_model=BatchForecastResponse, tags=["forecast"])
def predict_batch(req: BatchForecastRequest) -> BatchForecastResponse:
    started = time.perf_counter()
    try:
        return BatchForecastResponse(results=[_forecast_one(item) for item in req.items])
    finally:
        PREDICT_LATENCY.labels(endpoint="batch").observe(time.perf_counter() - started)


@app.get("/model", tags=["ops"])
def model_info() -> dict[str, object]:
    """What is actually serving right now - the first question in any incident."""
    bundle = predictor.load_models()
    heads = {
        name: {"num_trees": model.num_trees(), "n_features": model.num_feature()}
        for name, model in bundle.models.items()
    }
    return {"version": bundle.version, "heads": heads, "horizon_max": 28}
