"""Naming for the two quantile heads.

Both heads are separate LightGBM boosters, so they are separate registered models:
`<base>-q50` and `<base>-q90`. Registering only the median head - and then loading
that same URI twice at scoring time - silently makes P90 equal to P50, which zeroes
the safety stock and quietly turns the whole quantile design into a point forecast.

One module owns the naming so training, batch scoring and the API cannot disagree.
"""

from __future__ import annotations

# label used in code and responses -> quantile suffix used in every artefact name
HEADS: dict[str, str] = {"p50": "q50", "p90": "q90"}


def head_for(quantile: float) -> str:
    """0.5 -> 'q50'. The suffix that names this quantile everywhere."""
    return f"q{int(round(quantile * 100))}"


def artifact_path(head: str) -> str:
    """MLflow artifact path inside a training run, e.g. 'model_q50'."""
    return f"model_{head}"


def registered_name(base: str, head: str) -> str:
    """Registered model name for one head, e.g. 'shelfcast-demand-forecaster-q50'."""
    return f"{base}-{head}"


def head_uri(base: str, head: str, stage: str = "Production") -> str:
    """Registry URI for one head, e.g. 'models:/shelfcast-demand-forecaster-q50/Production'."""
    return f"models:/{registered_name(base, head)}/{stage}"
