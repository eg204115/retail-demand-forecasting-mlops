# Shelfcast — retail demand forecasting, end to end

SKU-level daily demand forecasting for a supermarket chain, wired into the thing a
forecast is actually for: a replenishment order. LightGBM quantile models (P50 and
P90) served two ways, behind a pipeline where every promotion is gated and every
number is reproducible.

The model is the least interesting part of this repository. What makes it MLOps
rather than a notebook is that ingestion, feature engineering, training, evaluation,
promotion, serving and monitoring are all automated, versioned and tested — and that
a model which cannot beat seasonal naive does not ship.

---

## The loop

```
raw CSV -> bronze parquet -> PySpark features -> Feast -> LightGBM P50/P90 -> MLflow
                  |                  |                            |             |
            data contracts     leakage guards              backtest gate    registry
                                                                                |
                                          batch order plan  <-  Production  ->  FastAPI
                                                                                |
                                                    Evidently drift + Prometheus/Grafana
                                                                                |
                                                          gate trips -> retrain (same path)
```

Full diagram and rationale: [docs/architecture.md](docs/architecture.md).

---

## Quickstart

Requires Python 3.10 or 3.11, Java 17 (for PySpark) and Docker.

```bash
make setup                 # editable install with all extras + pre-commit hooks
make stack-up              # MLflow, MinIO, Redis, Postgres, Prometheus, Grafana
shelfcast ingest --sample  # synthetic data with the M5 schema, no Kaggle account needed
shelfcast bronze
shelfcast features
shelfcast train
shelfcast backtest         # exits non-zero if the candidate loses to the baseline
shelfcast promote          # only promotes if the gate above passed
shelfcast serve
```

Then:

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"store_id":"CA_1","item_id":"ITEM_001","horizon":14}'
```

| Service | URL |
|---|---|
| Forecast API + OpenAPI docs | http://localhost:8000/docs |
| MLflow tracking + registry | http://localhost:5000 |
| Grafana (serving dashboard) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| MinIO console | http://localhost:9001 |

For the real dataset, put Kaggle credentials in `~/.kaggle/kaggle.json` and drop the
`--sample` flag. Every command is also a Make target (`make train`) and a module
(`python -m src.training.train`) — same code path either way.

### Watch the monitoring actually fire

```bash
make drift-demo
```

Injects a chain-wide promotion (prices down, volume up) into a copy of the feature
table, then runs the drift check against it. The gate trips, writes
`artifacts/reports/drift_report.html`, and exits non-zero — which is what triggers
retraining in CI. A monitoring stack nobody has watched fire is a config file, not a
control.

---

## What is in here

| Path | What it holds |
|---|---|
| `src/ingestion/` | Download, and the wide-to-long unpivot into partitioned Parquet |
| `src/features/` | PySpark feature engineering, Spark tuning, Feast definitions |
| `src/training/` | Dataset splits, baselines, training, backtest gate, registry, inventory policy |
| `src/serving/` | FastAPI service, batch scoring, online feature lookup |
| `src/monitoring/` | Evidently drift + performance checks, and the drift injector |
| `pipelines/airflow/` | Training and monitoring DAGs |
| `tests/` | Unit, Spark, data-contract and API tests |
| `infra/terraform/` | ECR + Fargate serving infrastructure |
| `docs/` | Architecture, model card, ADRs, Spark optimisation log |

---

## Engineering decisions worth defending

**Quantile regression, not point forecasts.** The consumer is an inventory policy
that needs a service level. Safety stock is the gap between the P90 and the median,
scaled over the protection interval — a conditional mean cannot size it. Both heads
are registered and promoted together (`<model>-q50`, `<model>-q90`), because a P50
from one run serving alongside a P90 from another silently corrupts that gap.

**Leakage is designed against, not hoped against.** A feature for date *d* may only
read data from *d − horizon*. The floor is applied once, in
`transforms.add_lag_features`, every rolling window is computed on already-lagged
values, and splits are chronological with a horizon-sized gap. Both rules are
asserted in CI — see `test_lag_respects_the_forecast_horizon`.

**The promotion gate is the point.** A rolling-origin backtest refits per fold and
compares the candidate against seasonal naive *and* the incumbent. Losing candidates
do not ship, the verdict is written as JSON so CI never parses logs, and a missing
verdict fails closed. Forcing a promotion requires a written reason.

**A feature store for one model.** It looks like over-engineering until the third
consumer. The real justification is the point-in-time join: `get_historical_features`
rewinds each feature to what was known at the label's timestamp, and the same
definitions serve the online store, so training and serving compute features from one
source of truth. Category encodings ship in the model metadata for the same reason —
a pandas category is encoded by position, so rebuilding levels at serving time would
give the same store a different code than it had in training.

**Batch is the primary serving path.** Replenishment is a nightly decision. The API
exists for what-if queries and the store app; making it primary would invent latency
requirements the business does not have.

**LightGBM over a deep temporal model.** ~30k series with strong tabular structure is
where GBDTs still win, and minutes-per-fit decides how many experiments fit in a
week. Reasoning, and the conditions under which it flips:
[ADR 0002](docs/adr/0002-lightgbm-over-deep-temporal-models.md).

**PySpark tuned with numbers, not vibes.** `stack()` instead of a union loop,
broadcast the small side, one repartition before ~20 window functions, AQE plus
targeted salting for the two hot stores. Each change and its measured effect:
[docs/spark-optimisation.md](docs/spark-optimisation.md).

---

## Metrics

WMAPE is primary: retail demand is intermittent, MAPE is undefined on zero-sales
days, and weighting error by volume matches the business cost. Reported alongside it:
RMSE, MAE, signed bias (persistent positive bias means systematic over-ordering),
pinball loss per quantile, and P90 coverage.

Every evaluation reports seasonal naive on the same window. Fill these in from
`artifacts/reports/backtest.json` after a full run on real data:

| Metric | Seasonal naive | Model | Lift |
|---|---|---|---|
| WMAPE | _TBD_ | _TBD_ | _TBD_ |
| RMSE | _TBD_ | _TBD_ | _TBD_ |
| P90 coverage | n/a | _TBD_ | target 0.90 |

Scope, limitations, and the cases this model must not be used for:
[docs/model-card.md](docs/model-card.md).

---

## Testing

```bash
make test        # everything, including the Spark tests
make test-fast   # skips JVM startup and the integration tests
make lint        # ruff + mypy
```

Four layers, each catching a different failure:

- **Unit** — metrics, config, splits, inventory policy.
- **Spark** — the leakage guarantees, on a real local SparkSession.
- **Data contracts** — pandera schemas that run as a *pipeline step*, not just a
  test, so a schema break fails a DAG task instead of training a model on nulls.
- **API** — the HTTP contract against a stubbed booster: quantile ordering, 422s on
  bad input, 503 when no model is loaded.

CI additionally runs the whole pipeline end to end on synthetic data, so a break in
the wiring between stages fails a pull request.

---

## Operations

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR | Lint, tests, end-to-end smoke run, image build |
| `train.yml` | Mondays 02:00 UTC | Retrain, backtest gate, promote or fail loudly |
| `drift-check.yml` | daily 06:00 UTC | Drift report; opens an issue and triggers retraining |
| `deploy.yml` | push to serving code | Build, smoke-test the image, ECS rolling deploy with rollback |

Images are tagged by commit SHA (never `:latest`) so a rollback has something to
address. `/health` is shallow, `/ready` scores a real row — a readiness probe that
only checks the process keeps routing traffic to a container with no model.

## Licence

MIT — see [LICENSE](LICENSE).
