"""Model loading and feature retrieval for online scoring.

Features come from the Feast online store using the definitions the training set was
built from, which is the whole reason the feature store exists: the same transform
code produces the training row and the serving row.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.common import HEADS, get_logger, head_uri
from src.training.dataset import apply_category_levels

log = get_logger(__name__)

DEFAULT_MODEL_NAME = "shelfcast-demand-forecaster"
DEFAULT_STAGE = "Production"

ONLINE_FEATURES = [
    "demand_features:lag_1",
    "demand_features:lag_7",
    "demand_features:lag_14",
    "demand_features:lag_28",
    "demand_features:roll_mean_7",
    "demand_features:roll_mean_28",
    "demand_features:roll_mean_91",
    "demand_features:roll_std_7",
    "demand_features:roll_std_28",
    "demand_features:zero_share_28",
    "demand_features:sell_price",
    "demand_features:price_ratio",
    "demand_features:is_discounted",
    "demand_features:is_event",
]


class ModelBundle:
    """The P50 and P90 boosters plus the version string that produced them."""

    def __init__(
        self,
        models: dict[str, Any],
        version: str,
        categoricals: dict[str, list[str]] | None = None,
    ) -> None:
        self.models = models
        self.version = version
        # Category levels recorded at training time; empty when the model predates
        # them, in which case scoring falls back to per-request categories.
        self.categoricals = categoricals or {}

    @property
    def loaded(self) -> bool:
        return bool(self.models)

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            name: np.clip(model.predict(align_to_model(frame, model, self.categoricals)), 0, None)
            for name, model in self.models.items()
        }


def load_head(uri: str) -> tuple[Any, dict]:
    """Load one booster and whatever metadata was logged alongside it."""
    import mlflow.lightgbm
    from mlflow.models import get_model_info

    model = mlflow.lightgbm.load_model(uri)
    try:
        metadata = get_model_info(uri).metadata or {}
    except Exception as exc:  # older runs carry no metadata; not fatal
        log.warning("no model metadata for %s: %s", uri, exc)
        metadata = {}
    return model, metadata


@lru_cache(maxsize=1)
def load_models(name: str | None = None, stage: str | None = None) -> ModelBundle:
    """Load both heads from the MLflow registry once per process.

    Cached because a cold load is ~200ms; on a container with a readiness probe that
    is the difference between a fast rollout and a flapping one.

    Each quantile is its own registered model (`<name>-q50`, `<name>-q90`). Loading
    one URI for both heads would make P90 equal P50 and silently zero the safety
    stock, so the two are addressed separately and by name.
    """
    base = name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    stage = stage or os.getenv("MODEL_STAGE", DEFAULT_STAGE)

    models: dict[str, Any] = {}
    categoricals: dict[str, list[str]] = {}
    for label, head in HEADS.items():
        # MODEL_URI_P50 / MODEL_URI_P90 override the registry for local runs.
        uri = os.getenv(f"MODEL_URI_{label.upper()}") or head_uri(base, head, stage)
        try:
            model, metadata = load_head(uri)
        except Exception as exc:
            log.error("failed to load %s from %s: %s", label, uri, exc)
            continue
        models[label] = model
        categoricals = categoricals or dict(metadata.get("categoricals") or {})

    version = os.getenv("MODEL_VERSION", f"{base}/{stage}")
    log.info("loaded %d model heads for %s/%s", len(models), base, stage)
    return ModelBundle(models, version, categoricals)


@lru_cache(maxsize=1)
def get_feature_store():
    """Feast client, or None when the online store is unreachable."""
    try:
        from feast import FeatureStore

        return FeatureStore(repo_path=os.getenv("FEAST_REPO", "conf"))
    except Exception as exc:
        log.warning("feature store unavailable, falling back to request features: %s", exc)
        return None


def fetch_online_features(store_id: str, item_id: str) -> dict[str, float]:
    fs = get_feature_store()
    if fs is None:
        return {}
    rows = fs.get_online_features(
        features=ONLINE_FEATURES,
        entity_rows=[{"series_id": f"{store_id}_{item_id}"}],
    ).to_dict()
    return {
        k: (v[0] if v and v[0] is not None else 0.0) for k, v in rows.items() if k != "series_id"
    }


def build_frame(
    store_id: str,
    item_id: str,
    origin: date,
    horizon: int,
    overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One row per forecast day: static series features + that day's calendar features."""
    features = fetch_online_features(store_id, item_id)
    if overrides:
        features.update(overrides)

    rows = []
    for step in range(1, horizon + 1):
        target = origin + timedelta(days=step)
        # Spark's dayofweek is 1=Sunday; the calendar features must be encoded the
        # same way here or every day-of-week feature is off by one against training.
        dow = target.isoweekday() % 7 + 1
        row = dict(features)
        row.update(
            {
                "store_id": store_id,
                "item_id": item_id,
                "target_date": target,
                "dow": dow,
                "day_of_month": target.day,
                "week_of_year": target.isocalendar().week,
                "month": target.month,
                "year": target.year,
                "is_weekend": int(target.isoweekday() >= 6),
                "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
                "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
                "horizon_step": step,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def align_to_model(
    frame: pd.DataFrame,
    model: Any,
    categoricals: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Reindex to the exact training column order; missing columns become NaN.

    LightGBM will happily score a mis-ordered frame and return nonsense, so the
    alignment is explicit rather than trusted, and categoricals are re-encoded with
    the training levels rather than with whatever this request contained.
    """
    if categoricals:
        frame = apply_category_levels(frame, categoricals)
    else:
        frame = frame.copy()
        for col in ("store_id", "item_id"):
            if col in frame.columns:
                frame[col] = frame[col].astype("category")

    expected = list(model.feature_name())
    return frame.reindex(columns=expected)
