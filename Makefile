.PHONY: help setup lint format test test-fast features train evaluate serve \
        stack-up stack-down docker-build drift demo clean

PY := python
CONF := conf/config.yaml

help:  ## Show available targets
	grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package with all extras + dev tooling, and pre-commit hooks
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[spark,featurestore,serving,monitoring,dev]"
	pre-commit install

lint:  ## Ruff + mypy
	ruff check src tests
	ruff format --check src tests
	mypy src

format:  ## Auto-format
	ruff check --fix src tests
	ruff format src tests

test:  ## Full test suite (includes slow Spark tests)
	pytest --cov=src --cov-report=term-missing --cov-report=xml

test-fast:  ## Skip Spark and integration tests
	pytest -m "not spark and not integration"

ingest:  ## Download the raw dataset into data/raw and land it as bronze parquet
	$(PY) -m src.ingestion.download --config $(CONF)
	$(PY) -m src.ingestion.to_bronze --config $(CONF)

features:  ## Build the PySpark feature table (silver -> features) and materialise to Feast
	$(PY) -m src.features.build_features --config $(CONF)
	cd conf && feast apply && cd ..

train:  ## Train the model, log run + params + metrics to MLflow
	$(PY) -m src.training.train --config $(CONF)

evaluate:  ## Backtest the candidate against production; gate promotion
	$(PY) -m src.training.evaluate --config $(CONF)

promote:  ## Promote the latest candidate to Production if it clears the gate
	$(PY) -m src.training.registry promote --config $(CONF)

serve:  ## Run the FastAPI scoring service locally
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

batch-score:  ## Score the next horizon for every store x SKU and write the order plan
	$(PY) -m src.serving.batch_score --config $(CONF)

drift:  ## Run the Evidently drift report against the reference window
	$(PY) -m src.monitoring.drift --config $(CONF)

drift-demo:  ## Inject a promo shift into serving data and show monitoring catch it
	$(PY) -m src.monitoring.inject_drift --config $(CONF)
	$(PY) -m src.monitoring.drift --config $(CONF) --input data/features/sales_features_drifted \
		|| echo ">>> drift gate tripped as expected (this is the demo working)"

stack-up:  ## Start MLflow, MinIO, Redis, Postgres, Prometheus, Grafana
	docker compose up -d
	echo "MLflow    http://localhost:5000"
	echo "MinIO     http://localhost:9001"
	echo "Grafana   http://localhost:3000"

stack-down:  ## Stop the local stack
	docker compose down -v

docker-build:  ## Build the training and serving images
	docker build -f docker/Dockerfile.serving -t shelfcast-serving:local .
	docker build -f docker/Dockerfile.training -t shelfcast-training:local .

demo: stack-up ingest features train evaluate  ## One-command end-to-end run
	$(MAKE) serve

clean:  ## Remove caches and local artifacts (leaves raw data alone)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	rm -rf artifacts/reports/* data/features/* data/silver/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
