"""The promotion gate is the one control that decides what production serves.

These tests are about the failure direction: a gate that cannot read its verdict must
block, never wave the candidate through.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mlflow")

from src.common.config import Config  # noqa: E402
from src.training.registry import gate_passed  # noqa: E402


def _cfg(reports) -> Config:
    return Config({"paths": {"reports": str(reports)}})


def test_a_passing_verdict_clears_the_gate(tmp_path):
    (tmp_path / "backtest.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    passed, reasons = gate_passed(_cfg(tmp_path))
    assert passed and reasons == []


def test_a_failing_verdict_blocks_and_reports_why(tmp_path):
    verdict = {"passed": False, "reasons": ["candidate wmape 0.42 does not beat baseline 0.30"]}
    (tmp_path / "backtest.json").write_text(json.dumps(verdict), encoding="utf-8")
    passed, reasons = gate_passed(_cfg(tmp_path))
    assert not passed
    assert "baseline" in reasons[0]


def test_a_missing_verdict_fails_closed(tmp_path):
    """ "We did not check" is not "it passed".

    A skipped backtest step, a wiped artifacts directory or a job that died before
    writing its verdict all land here, and all of them must stop the promotion.
    """
    passed, reasons = gate_passed(_cfg(tmp_path))
    assert not passed
    assert reasons and "no backtest verdict" in reasons[0]
