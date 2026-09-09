"""Weekly retraining pipeline.

Ordering matters and is enforced by the DAG rather than by convention: features are
validated before training, the model is backtested before it is registered, and it
is registered before it is promoted. The promotion task is the only one allowed to
change what production serves.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "octave-ml",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "depends_on_past": False,
}

CONF = "conf/config.yaml"

with DAG(
    dag_id="shelfcast_training",
    description="Ingest -> features -> train -> backtest -> register -> promote",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * 1",  # Mondays 02:00, after the weekend sales land
    catchup=False,
    max_active_runs=1,
    tags=["ml", "forecasting", "retail"],
) as dag:
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"python -m src.ingestion.to_bronze --config {CONF}",
    )

    validate_raw = BashOperator(
        task_id="validate_raw",
        bash_command="python -m tests.data_quality.validate --layer bronze",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"python -m src.features.build_features --config {CONF}",
        execution_timeout=timedelta(hours=2),
    )

    materialise = BashOperator(
        task_id="materialise_online_features",
        bash_command="cd conf && feast materialize-incremental $(date -u +'%Y-%m-%dT%H:%M:%S')",
    )

    train = BashOperator(
        task_id="train",
        bash_command=f"python -m src.training.train --config {CONF}",
        execution_timeout=timedelta(hours=3),
    )

    # Non-zero exit here is the promotion gate refusing the candidate, which should
    # fail the run loudly rather than be swallowed.
    backtest = BashOperator(
        task_id="backtest_gate",
        bash_command=f"python -m src.training.evaluate --config {CONF}",
    )

    promote = BashOperator(
        task_id="promote",
        bash_command=f"python -m src.training.registry promote --config {CONF}",
    )

    batch_score = BashOperator(
        task_id="batch_score",
        bash_command=f"python -m src.serving.batch_score --config {CONF}",
    )

    def _notify(**context):
        run = context["dag_run"]
        print(f"shelfcast_training {run.run_id} finished at {context['ts']}")

    notify = PythonOperator(task_id="notify", python_callable=_notify)

    ingest >> validate_raw >> build_features >> materialise >> train >> backtest >> promote >> batch_score >> notify
