"""The drift gate's contract with CI.

The gate communicates through three channels - its exit code, the summary file, and
its GitHub step outputs - and the workflow reads all three. These tests pin that
contract, because the failure it replaced was silent: a run that crashed before
writing anything exited with the same code as a run that detected drift, and the
workflow read the crash as a verdict and went looking for a file that did not exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.config import Config
from src.monitoring import drift as drift_module

WINDOW_DAYS = 28


def _cfg(tmp_path: Path) -> Config:
    return Config(
        {
            "paths": {
                "features": str(tmp_path / "features"),
                "reports": str(tmp_path / "reports"),
            },
            "monitoring": {
                "reference_window_days": WINDOW_DAYS,
                "drift_share_threshold": 0.3,
                "wmape_degradation_threshold": 0.15,
            },
        }
    )


def _panel(days: int) -> pd.DataFrame:
    """A feature panel spanning `days` days, enough to fill both windows when large."""
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "units": range(days),
            "lag_1": range(days),
            "sell_price": [3.0] * days,
        }
    )


def _evidently_payload(share: float) -> dict:
    """The shape `summarise` expects back from Evidently."""
    return {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "number_of_columns": 10,
                    "number_of_drifted_columns": int(share * 10),
                    "share_of_drifted_columns": share,
                    "drift_by_columns": {
                        "lag_1": {"drift_detected": share > 0},
                        "sell_price": {"drift_detected": False},
                    },
                },
            }
        ]
    }


@pytest.fixture
def gh_output(tmp_path, monkeypatch) -> Path:
    """Point GITHUB_OUTPUT at a file, as the Actions runner does."""
    path = tmp_path / "gh_output"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(path))
    return path


@pytest.fixture
def configured(tmp_path, monkeypatch) -> Config:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(drift_module, "load_config", lambda _path: cfg)
    return cfg


def _run(argv: list[str] | None = None) -> int:
    """Run main() and return the exit code it raised."""
    with pytest.raises(SystemExit) as excinfo:
        drift_module.main(argv or ["--config", "unused.yaml"])
    return excinfo.value.code


def _stub_features(cfg: Config, monkeypatch, days: int) -> None:
    """Make the feature table exist and read back as a panel, without needing parquet."""
    table = Path(cfg.paths.features) / "sales_features"
    table.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(drift_module.pd, "read_parquet", lambda _p: _panel(days))


class TestFailureIsNotAVerdict:
    """A check that could not run must be distinguishable from one that found drift."""

    def test_missing_feature_table_exits_error_not_gate_tripped(self, configured):
        assert _run() == drift_module.EXIT_ERROR

    def test_missing_feature_table_writes_no_summary(self, configured):
        _run()
        assert not (Path(configured.paths.reports) / "drift_summary.json").exists()

    def test_missing_feature_table_emits_no_outputs(self, configured, gh_output):
        """The regression test.

        The workflow branches on `retrain_recommended`. When the check dies before
        reaching a verdict there must be no output at all, so the workflow sees an
        empty string and fails the job instead of opening an issue about a summary
        file that was never written.
        """
        _run()
        assert gh_output.read_text(encoding="utf-8") == ""

    def test_report_directory_exists_even_when_the_run_fails(self, configured):
        """An empty directory says "nothing to report"; a missing one is a stack trace."""
        _run()
        assert Path(configured.paths.reports).is_dir()

    def test_insufficient_history_exits_error(self, configured, monkeypatch, gh_output):
        _stub_features(configured, monkeypatch, days=3)
        assert _run() == drift_module.EXIT_ERROR
        assert gh_output.read_text(encoding="utf-8") == ""


class TestVerdicts:
    """A run that completes reports through its exit code, its summary, and its outputs."""

    @pytest.fixture(autouse=True)
    def _features(self, configured, monkeypatch):
        _stub_features(configured, monkeypatch, days=WINDOW_DAYS * 3)

    def _with_drift_share(self, monkeypatch, share: float) -> None:
        monkeypatch.setattr(
            drift_module,
            "evidently_report",
            lambda reference, current, out_dir: _evidently_payload(share),
        )

    def test_gate_trips_on_drift_above_the_threshold(self, monkeypatch, configured, gh_output):
        self._with_drift_share(monkeypatch, 0.5)
        assert _run() == drift_module.EXIT_GATE_TRIPPED

        summary = json.loads(
            (Path(configured.paths.reports) / "drift_summary.json").read_text(encoding="utf-8")
        )
        assert summary["retrain_recommended"] is True
        assert summary["drift_share"] == 0.5
        assert "retrain_recommended=true" in gh_output.read_text(encoding="utf-8")

    def test_clean_run_exits_ok_and_says_so(self, monkeypatch, configured, gh_output):
        self._with_drift_share(monkeypatch, 0.1)
        assert _run() == drift_module.EXIT_OK

        summary = json.loads(
            (Path(configured.paths.reports) / "drift_summary.json").read_text(encoding="utf-8")
        )
        assert summary["retrain_recommended"] is False
        assert "retrain_recommended=false" in gh_output.read_text(encoding="utf-8")

    def test_the_three_exit_codes_are_distinct(self):
        codes = {
            drift_module.EXIT_OK,
            drift_module.EXIT_ERROR,
            drift_module.EXIT_GATE_TRIPPED,
        }
        assert len(codes) == 3


class TestEmitGithubOutput:
    def test_writes_key_value_pairs(self, gh_output):
        drift_module.emit_github_output(a="1", b="2")
        assert gh_output.read_text(encoding="utf-8") == "a=1\nb=2\n"

    def test_appends_rather_than_replacing(self, gh_output):
        drift_module.emit_github_output(a="1")
        drift_module.emit_github_output(b="2")
        assert gh_output.read_text(encoding="utf-8") == "a=1\nb=2\n"

    def test_is_a_no_op_off_the_runner(self, monkeypatch):
        """Local runs and the Makefile demo have no GITHUB_OUTPUT; that is not an error."""
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        drift_module.emit_github_output(a="1")  # must not raise
