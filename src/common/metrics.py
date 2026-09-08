"""Forecast accuracy metrics.

WMAPE is the primary metric: intermittent retail demand means a lot of zero-sales
days, and MAPE is undefined on those. Weighting the absolute error by volume also
matches the business cost - being wrong on a fast mover matters more.
"""

from __future__ import annotations

import numpy as np


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denominator)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Signed error - a persistently positive bias means systematic over-ordering."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float((y_pred - y_true).sum() / denominator)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Loss the quantile models are actually optimising - report it, don't just train on it."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    delta = y_true - y_pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def coverage(y_true: np.ndarray, y_pred_upper: np.ndarray) -> float:
    """Share of actuals at or below the upper quantile - should sit near the nominal level."""
    return float(np.mean(np.asarray(y_true, float) <= np.asarray(y_pred_upper, float)))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "wmape": wmape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
