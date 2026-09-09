from __future__ import annotations

import numpy as np
import pytest

from src.training.inventory import order_up_to_level, protection_interval


def test_protection_interval_covers_lead_time_and_review_period():
    assert protection_interval(3, 7) == 10


def test_order_level_grows_with_forecast_uncertainty():
    tight = order_up_to_level([10.0], [11.0], 3, 7)
    wide = order_up_to_level([10.0], [20.0], 3, 7)
    assert wide[0] > tight[0]


def test_safety_stock_is_never_negative_when_quantiles_cross():
    # A crossed pair (P90 below P50) is a model bug, but it must not produce an
    # order below expected demand.
    level = order_up_to_level([10.0], [8.0], 3, 7)
    assert level[0] == pytest.approx(100.0)


def test_expected_demand_scales_linearly_with_the_interval():
    short = order_up_to_level([10.0], [10.0], 1, 1)
    long = order_up_to_level([10.0], [10.0], 5, 5)
    assert long[0] == pytest.approx(short[0] * 5)


def test_handles_vectors():
    level = order_up_to_level(np.array([1.0, 2.0]), np.array([2.0, 4.0]), 2, 2)
    assert level.shape == (2,)
    assert (level > 0).all()
