# Architecture

```
                       ┌───────────────────────────────────────────────┐
                       │              Orchestration (Airflow)          │
                       │  shelfcast_training  |  shelfcast_monitoring  │
                       └───────────────────────────────────────────────┘
                                          │ drives
   ┌──────────┐   ┌──────────┐   ┌────────▼────────┐   ┌──────────────┐
   │  Raw     │──▶│  Bronze  │──▶│  Feature build  │──▶│ Feature table│
   │  CSV     │   │ Parquet  │   │    (PySpark)    │   │  (Parquet)   │
   └──────────┘   └──────────┘   └─────────────────┘   └──────┬───────┘
        │              │                                      │
        │         data contracts                    ┌─────────┴─────────┐
        │        (pandera, per layer)               │       Feast       │
        │                                           │ offline  │ online │
        │                                           └────┬─────┴────┬───┘
        │                                                │          │
        │                            point-in-time join  │          │ low-latency
        │                                                ▼          ▼
        │                                        ┌──────────────┐  ┌──────────────┐
        │                                        │   Training   │  │   Serving    │
        │                                        │  LightGBM    │  │   FastAPI    │
        │                                        │  P50 + P90   │  │  + batch     │
        │                                        └──────┬───────┘  └──────┬───────┘
        │                                               │                 │
        │                                    ┌──────────▼──────────┐      │
        │                                    │  MLflow: tracking,  │      │
        │                                    │  registry, staging  │◀─────┘
        │                                    └──────────┬──────────┘
        │                                               │ promotion gate
        │                                    ┌──────────▼──────────┐
        └───────────────────────────────────▶│  Monitoring         │
                     actuals land            │  Evidently drift    │
                                             │  Prometheus/Grafana │
                                             └──────────┬──────────┘
                                                        │ gate trips
                                                        └──▶ retrain
```

## The loop, not the model

The model is the least interesting part of this repository. What makes it an MLOps
project rather than a notebook is that every arrow above is automated and every
promotion is gated:

1. **Ingestion** lands raw CSV as partitioned Parquet, reshaped once from wide to
   long so nothing downstream repeats that work.
2. **Data contracts** (`tests/data_quality/`) run as a pipeline step, not as a test
   someone remembers to run. A schema break fails a task instead of training a model
   on nulls.
3. **Feature engineering** is PySpark, with the horizon floor that prevents leakage
   applied in one place and asserted in CI.
4. **The feature store** serves the same definitions to training (point-in-time
   correct historical join) and to serving (online lookup), which is the only
   structural defence against training/serving skew.
5. **Training** logs every parameter, metric and artifact to MLflow, including the
   category encodings the boosters were fitted with, so serving reproduces them
   exactly. A run is a record, not a side effect of someone's terminal.
6. **The promotion gate** backtests the candidate on rolling origins against both a
   seasonal-naive baseline and the incumbent. Losing candidates do not ship.
7. **Serving** is the same registered model in two shapes: a nightly batch job that
   produces the order plan, and a low-latency API for what-if queries.
8. **Monitoring** watches inputs (immediately) and accuracy (once actuals land).
   Sustained degradation triggers retraining through the same gated path - there is
   exactly one way for a model to reach production.

## Design decisions worth defending

See `docs/adr/` for the full reasoning. In brief:

- **LightGBM over a deep temporal model.** 30k series with strong tabular structure
  is where GBDTs still win, and a global model trains in minutes rather than hours.
  A TFT branch exists for comparison, not as the default.
- **Quantile regression, not point forecasts.** The consumer is an inventory policy
  that needs a service level. A conditional mean cannot size safety stock.
- **A feature store for a single-model project.** It looks like over-engineering
  until the third consumer appears. The real justification is the point-in-time join.
- **Batch as the primary serving path.** Replenishment is a nightly decision. The
  API exists for interactive queries; making it the primary path would add latency
  requirements the business does not have.
