"""Daily monitoring, with a conditional retrain trigger.

Runs every morning. If the drift gate trips it triggers the training DAG rather than
retraining inline, so there is exactly one code path that can produce a production
model and one place to look when something ships.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULT_ARGS = {
    "owner": "octave-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def run_drift_check(**context) -> bool:
    """SystemExit(1) from the drift module means the gate tripped, not that it crashed."""
    import json
    import subprocess
    from pathlib import Path

    subprocess.run(
        ["python", "-m", "src.monitoring.drift", "--config", "conf/config.yaml"],
        check=False,
    )
    summary = json.loads(Path("artifacts/reports/drift_summary.json").read_text(encoding="utf-8"))
    context["ti"].xcom_push(key="drift_summary", value=summary)
    return bool(summary.get("retrain_recommended"))


def choose_branch(**context) -> str:
    tripped = context["ti"].xcom_pull(task_ids="drift_check")
    return "trigger_retraining" if tripped else "no_action"


with DAG(
    dag_id="shelfcast_monitoring",
    description="Daily drift + performance check; triggers retraining when it trips",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="0 6 * * *",
    catchup=False,
    tags=["ml", "monitoring"],
) as dag:
    drift_check = PythonOperator(task_id="drift_check", python_callable=run_drift_check)
    branch = BranchPythonOperator(task_id="branch", python_callable=choose_branch)

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="shelfcast_training",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    no_action = PythonOperator(
        task_id="no_action",
        python_callable=lambda **_: print("no drift detected"),
    )

    drift_check >> branch >> [trigger_retraining, no_action]
