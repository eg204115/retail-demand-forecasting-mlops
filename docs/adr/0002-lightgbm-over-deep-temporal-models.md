# ADR 0002: LightGBM as the default model, not a deep temporal model

**Status:** Accepted · **Date:** 2026-09-07

## Context

The obvious modern choice for multi-horizon forecasting is a deep model - Temporal
Fusion Transformer, N-BEATS, DeepAR. The dataset is ~30,000 daily series with strong
tabular structure: calendar effects, price, promotions, intermittency.

## Decision

LightGBM with quantile objectives is the default. A PyTorch temporal model is kept
as an optional comparison branch (`.[deep]`), not as the production path.

## Rationale

- **Empirical.** On M5, gradient-boosted trees on well-built lag features were the
  competitive approach; the winning solutions were LightGBM. On tabular retail data
  with strong exogenous structure, GBDTs remain hard to beat.
- **Iteration speed.** A full global fit is minutes on one machine, versus hours on a
  GPU. That difference decides how many experiments fit in a week, and experiment
  count is what actually improves a forecast.
- **Operational cost.** No GPU in the serving path. A LightGBM booster scores in
  microseconds on CPU and the container is small.
- **Debuggability.** Feature importance and per-split behaviour give a planner
  something to argue with. "The transformer says 40" is not a conversation.
- **Intermittency.** Quantile GBDTs handle zero-heavy targets without the
  distributional assumptions a DeepAR-style likelihood head imposes.

## Consequences

Cross-series representation learning is left on the table; a deep model would likely
win on cold-start and on long-range seasonality. The comparison branch exists so that
claim can be tested rather than assumed. If the TFT branch wins the backtest, this
ADR gets superseded.
