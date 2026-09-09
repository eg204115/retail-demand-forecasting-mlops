from __future__ import annotations

import pytest

from src.common.config import Config, load_config, resolve


def test_attribute_access_is_nested():
    cfg = Config({"a": {"b": {"c": 1}}})
    assert cfg.a.b.c == 1


def test_missing_key_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = Config({}).nope


def test_resolve_returns_default_for_missing_path():
    assert resolve(Config({"a": {}}), "a.b.c", "fallback") == "fallback"


def test_env_vars_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELFCAST_TEST_URI", "http://tracking:5000")
    path = tmp_path / "c.yaml"
    path.write_text("mlflow:\n  tracking_uri: ${SHELFCAST_TEST_URI}\n", encoding="utf-8")
    assert load_config(path).mlflow.tracking_uri == "http://tracking:5000"


def test_default_is_used_when_the_var_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SHELFCAST_ABSENT", raising=False)
    path = tmp_path / "c.yaml"
    path.write_text("a: ${SHELFCAST_ABSENT:-fallback}\n", encoding="utf-8")
    assert load_config(path).a == "fallback"


def test_env_wins_over_the_default(tmp_path, monkeypatch):
    # This is what lets CI point the tracking URI at a file store without editing
    # conf/config.yaml, so the committed config and the CI run stay the same file.
    monkeypatch.setenv("SHELFCAST_PRESENT", "from-env")
    path = tmp_path / "c.yaml"
    path.write_text("a: ${SHELFCAST_PRESENT:-fallback}\n", encoding="utf-8")
    assert load_config(path).a == "from-env"


def test_unresolvable_reference_is_left_visible(tmp_path, monkeypatch):
    # Better a config that obviously did not resolve than one that quietly became "".
    monkeypatch.delenv("SHELFCAST_ABSENT", raising=False)
    path = tmp_path / "c.yaml"
    path.write_text("a: ${SHELFCAST_ABSENT}\n", encoding="utf-8")
    assert load_config(path).a == "${SHELFCAST_ABSENT}"


def test_tracking_uri_honours_the_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    assert load_config("conf/config.yaml").mlflow.tracking_uri == "file:./mlruns"


def test_real_config_has_the_keys_the_pipeline_reads(cfg):
    # Guards against a rename in config.yaml silently breaking a job at 2am.
    for dotted in (
        "paths.features",
        "data.horizon",
        "model.quantiles",
        "mlflow.registered_model",
        "promotion.primary_metric",
        "inventory.service_level",
        "monitoring.drift_share_threshold",
    ):
        assert resolve(cfg, dotted) is not None, dotted
