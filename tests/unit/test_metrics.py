from __future__ import annotations

import numpy as np
import pytest

from src.common.metrics import bias, coverage, evaluate_all, pinball_loss, wmape


def test_wmape_is_zero_for_perfect_forecast():
    y = np.array([1.0, 5.0, 3.0])
    assert wmape(y, y) == pytest.approx(0.0)


def test_wmape_is_nan_when_all_actuals_are_zero():
    # Intermittent SKUs really do go a whole window without a sale; the metric must
    # say "undefined" rather than divide by zero.
    assert np.isnan(wmape(np.zeros(5), np.ones(5)))


def test_wmape_weights_by_volume():
    y = np.array([100.0, 1.0])
    err_on_big = wmape(y, np.array([90.0, 1.0]))
    err_on_small = wmape(y, np.array([100.0, 11.0]))
    assert err_on_big == pytest.approx(err_on_small)


def test_bias_sign_distinguishes_over_and_under_forecasting():
    y = np.array([10.0, 10.0])
    assert bias(y, np.array([12.0, 12.0])) > 0
    assert bias(y, np.array([8.0, 8.0])) < 0


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_pinball_loss_is_minimised_at_the_true_quantile(q):
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, 20000)
    truth = np.quantile(y, q)
    at_truth = pinball_loss(y, np.full_like(y, truth), q)
    assert at_truth < pinball_loss(y, np.full_like(y, truth + 1.0), q)
    assert at_truth < pinball_loss(y, np.full_like(y, truth - 1.0), q)


def test_coverage_matches_nominal_level():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 10000)
    upper = np.full_like(y, np.quantile(y, 0.9))
    assert coverage(y, upper) == pytest.approx(0.9, abs=0.02)


def test_evaluate_all_returns_the_expected_keys():
    y = np.array([1.0, 2.0, 3.0])
    assert set(evaluate_all(y, y + 0.1)) == {"wmape", "rmse", "mae", "bias"}
