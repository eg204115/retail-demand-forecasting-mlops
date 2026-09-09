"""`shelfcast` - one entry point for every stage of the pipeline.

Each command delegates to the module that owns the step rather than reimplementing
it, so `shelfcast train` and `python -m src.training.train` are the same code path
and the Makefile, the Airflow DAGs and a terminal cannot drift apart.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SKU-level retail demand forecasting and replenishment.",
)

CONFIG = typer.Option("conf/config.yaml", "--config", "-c", help="Path to config.yaml")


def _conf(config: str) -> list[str]:
    return ["--config", config]


@app.command()
def ingest(
    config: str = CONFIG,
    sample: bool = typer.Option(False, help="Generate synthetic data instead of downloading"),
) -> None:
    """Fetch the raw dataset (or a synthetic stand-in) into data/raw."""
    from src.ingestion import download

    download.main(_conf(config) + (["--sample"] if sample else []))


@app.command()
def bronze(config: str = CONFIG) -> None:
    """Reshape the raw CSVs into partitioned, typed Parquet."""
    from src.ingestion import to_bronze

    to_bronze.main(_conf(config))


@app.command()
def features(
    config: str = CONFIG,
    explain: bool = typer.Option(False, help="Print the Spark physical plan and exit"),
) -> None:
    """Build the point-in-time-correct feature table."""
    from src.features import build_features

    build_features.main(_conf(config) + (["--explain"] if explain else []))


@app.command()
def validate(
    config: str = CONFIG,
    layer: str = typer.Option("features", help="Which data contract to enforce"),
) -> None:
    """Run the pandera data contract for one layer."""
    from tests.data_quality import validate as validator

    validator.main(_conf(config) + ["--layer", layer])


@app.command()
def train(
    config: str = CONFIG,
    run_name: str | None = typer.Option(None, "--run-name", help="MLflow run name"),
) -> None:
    """Fit both quantile heads and log the run to MLflow."""
    from src.training import train as trainer

    trainer.main(_conf(config) + (["--run-name", run_name] if run_name else []))


@app.command()
def backtest(config: str = CONFIG) -> None:
    """Rolling-origin backtest. Exits non-zero when the promotion gate fails."""
    from src.training import evaluate

    evaluate.main(_conf(config))


@app.command()
def promote(
    config: str = CONFIG,
    stage: str = typer.Option("Production", help="Target registry stage"),
    skip_gate: bool = typer.Option(False, "--skip-gate", help="Break-glass: ignore the gate"),
) -> None:
    """Register both heads and promote them, if the backtest gate cleared them."""
    from src.training import registry

    registry.main(
        ["promote", *_conf(config), "--stage", stage] + (["--skip-gate"] if skip_gate else [])
    )


@app.command()
def score(
    config: str = CONFIG,
    stage: str = typer.Option("Production", help="Registry stage to score with"),
    on_hand: str | None = typer.Option(None, "--on-hand", help="CSV of current stock"),
) -> None:
    """Batch-score every series and write the replenishment plan."""
    from src.serving import batch_score

    batch_score.main(
        _conf(config) + ["--stage", stage] + (["--on-hand", on_hand] if on_hand else [])
    )


@app.command()
def drift(
    config: str = CONFIG,
    input: str | None = typer.Option(None, "--input", help="Override the feature table path"),
) -> None:
    """Drift + performance check. Exits non-zero when retraining is recommended."""
    from src.monitoring import drift as drift_module

    drift_module.main(_conf(config) + (["--input", input] if input else []))


@app.command("inject-drift")
def inject_drift(
    config: str = CONFIG,
    window_days: int = typer.Option(14, help="How many recent days to perturb"),
    intensity: float = typer.Option(0.3, help="0.3 simulates a 30% price cut"),
) -> None:
    """Write a deliberately drifted copy of the feature table, for the demo."""
    from src.monitoring import inject_drift as injector

    injector.main(
        _conf(config) + ["--window-days", str(window_days), "--intensity", str(intensity)]
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Reload on code changes (development only)"),
) -> None:
    """Run the FastAPI scoring service."""
    import uvicorn

    uvicorn.run("src.serving.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
