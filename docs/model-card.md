# Model card: shelfcast-demand-forecaster

## Overview

| | |
|---|---|
| **Task** | Daily unit-sales forecasting per (store, SKU), 1-14 days ahead |
| **Type** | Gradient-boosted trees (LightGBM), two quantile heads: P50 and P90 |
| **Version** | See the MLflow model registry - this card describes the family, not one version |
| **Owner** | Nethmi Poornima |
| **Registry** | `models:/shelfcast-demand-forecaster-q50/Production` and `...-q90/Production` |

## Intended use

**In scope.** Replenishment planning for established SKUs with at least ~91 days of
sales history, at a daily grain, over a horizon of up to 14 days.

**Out of scope, and why:**

- **New SKUs.** Every informative feature is a lag or a rolling statistic. With no
  history the model falls back to category-level signal it was never trained to rely
  on. New products need a separate cold-start approach.
- **Horizons beyond 28 days.** Features are floored at the training horizon; asking
  for a longer horizon silently degrades to a near-flat forecast.
- **Promotional events unlike anything in training.** A first-ever chain-wide 50%
  promotion is extrapolation, not interpolation. This is exactly the case the drift
  monitor is there to catch.
- **Anything but replenishment.** These forecasts are not a demand-curve model and
  must not be used to set prices; `sell_price` is an input, and reading a causal
  price effect off a predictive model is a well-known way to be badly wrong.

Each quantile is a separate registered model, promoted together from a single
training run. Serving them from different runs would corrupt the P90 minus P50 gap
that sizes the safety stock, so the registry transition is all-or-nothing.

## Data

- **Source.** M5 Forecasting - Accuracy (Walmart, 2011-2016), a public stand-in for
  a supermarket transaction feed.
- **Grain.** One row per (store_id, item_id, date).
- **Features.** Horizon-floored lags (1, 7, 14, 28), rolling mean/std/max/zero-share
  (7, 28, 91), price level, price change and discount flags, calendar and event
  flags, SNAP indicators.
- **Target.** `units` sold that day.
- **Splits.** Chronological, with a `horizon`-day gap between train and validation.
  Reported metrics come from a 3-fold rolling-origin backtest.

## Leakage controls

The two failure modes this pipeline explicitly defends against, because both produce
excellent offline metrics and a useless production model:

1. **Unfloored lags.** A feature for date *d* may only read data from *d - horizon*
   or earlier. Enforced in `transforms.add_lag_features`, asserted in
   `tests/unit/test_transforms.py::test_lag_respects_the_forecast_horizon`.
2. **Random splits on a time series.** All splits are chronological with a gap.
   Asserted in `tests/unit/test_dataset.py`.

## Metrics

**Primary: WMAPE.** Retail demand is intermittent - many SKU-days are zero, where
MAPE is undefined. Weighting absolute error by volume also matches the business
cost: being wrong on a fast mover matters more than on a slow one.

**Secondary:** RMSE, MAE, signed bias (persistent positive bias means systematic
over-ordering), pinball loss per quantile, and P90 coverage (should sit near 0.90 -
if it does not, the safety stock is mis-sized whatever the point forecast does).

**Reference point.** Every evaluation reports seasonal-naive on the same window. A
model that cannot beat it is not promoted.

| Metric | Baseline (seasonal naive) | Model | Lift |
|---|---|---|---|
| WMAPE | _TBD_ | _TBD_ | _TBD_ |
| RMSE | _TBD_ | _TBD_ | _TBD_ |
| P90 coverage | n/a | _TBD_ | target 0.90 |

> Fill these from `artifacts/reports/backtest.json` after your first full run.

## Ethical and operational considerations

- **Feedback loop.** Forecasts drive orders, orders constrain shelf availability, and
  availability caps observed sales. Train on unconstrained demand where possible; a
  stockout censors the target and the model learns the stockout, not the demand.
- **Waste.** For perishables the P90 safety stock trades spoilage against
  availability. The service level is a business decision and lives in
  `conf/config.yaml`, not in the model.
- **Labour.** Order plans change staff workload. The system recommends; a planner
  approves.

## Maintenance

- **Retraining.** Weekly, plus on demand when the drift gate trips.
- **Monitoring.** Daily input-drift and performance checks (`src/monitoring/drift.py`).
- **Promotion.** Automatic only when the rolling-origin backtest beats both the
  seasonal-naive baseline and the incumbent production model.
- **Rollback.** Previous versions stay in the registry; ECS keeps the prior task
  definition and the deploy workflow rolls back on a failed health check.
