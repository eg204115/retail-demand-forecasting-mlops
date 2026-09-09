"""API contract tests against a stubbed model.

The point is the HTTP contract and the failure modes, not the numbers: a real model
is covered by the training tests, but a 200 with a null forecast would sail through
those and break the store app.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


class StubBooster:
    """Stands in for a LightGBM Booster."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)

    def feature_name(self):
        return ["lag_1", "lag_7", "roll_mean_7", "dow", "is_weekend"]

    def num_trees(self):
        return 10

    def num_feature(self):
        return 5


@pytest.fixture
def client(monkeypatch):
    from src.serving import app as app_module
    from src.serving import predictor

    bundle = predictor.ModelBundle({"p50": StubBooster(4.0), "p90": StubBooster(9.0)}, "stub-v1")
    monkeypatch.setattr(predictor, "load_models", lambda *a, **k: bundle)
    monkeypatch.setattr(predictor, "get_feature_store", lambda: None)
    monkeypatch.setattr(app_module.predictor, "load_models", lambda *a, **k: bundle)
    monkeypatch.setattr(app_module.predictor, "get_feature_store", lambda: None)
    return TestClient(app_module.app)


def test_health_reports_model_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["feature_store"] == "unavailable"


def test_predict_returns_one_row_per_horizon_day(client):
    res = client.post("/predict", json={"store_id": "CA_1", "item_id": "A", "horizon": 14})
    assert res.status_code == 200
    body = res.json()
    assert len(body["forecasts"]) == 14
    assert body["model_version"] == "stub-v1"


def test_quantiles_are_ordered(client):
    body = client.post("/predict", json={"store_id": "CA_1", "item_id": "A"}).json()
    assert all(day["p90"] >= day["p50"] for day in body["forecasts"])


def test_order_quantity_is_present_and_non_negative(client):
    body = client.post("/predict", json={"store_id": "CA_1", "item_id": "A"}).json()
    assert body["order_qty"] >= 0


def test_horizon_beyond_the_trained_range_is_rejected(client):
    res = client.post("/predict", json={"store_id": "CA_1", "item_id": "A", "horizon": 90})
    assert res.status_code == 422


def test_blank_identifiers_are_rejected(client):
    res = client.post("/predict", json={"store_id": "  ", "item_id": "A"})
    assert res.status_code == 422


def test_batch_endpoint_preserves_order(client):
    payload = {"items": [{"store_id": "CA_1", "item_id": f"A{i}"} for i in range(5)]}
    body = client.post("/predict/batch", json=payload).json()
    assert [r["item_id"] for r in body["results"]] == [f"A{i}" for i in range(5)]


def test_metrics_endpoint_is_scrapeable(client):
    text = client.get("/metrics").text
    assert "shelfcast_predictions_total" in text


def test_missing_model_returns_503(monkeypatch):
    from src.serving import app as app_module
    from src.serving import predictor

    empty = predictor.ModelBundle({}, "none")
    monkeypatch.setattr(app_module.predictor, "load_models", lambda *a, **k: empty)
    res = TestClient(app_module.app).post("/predict", json={"store_id": "CA_1", "item_id": "A"})
    assert res.status_code == 503
