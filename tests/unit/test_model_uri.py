"""Naming is load-bearing here.

The two quantile heads are separate registered models. If training, batch scoring
and the API ever disagree about what those are called, the most likely failure is
that P90 silently resolves to the P50 model - which zeroes the safety stock while
every endpoint keeps returning a plausible number.
"""

from __future__ import annotations

import pytest

from src.common.model_uri import HEADS, artifact_path, head_for, head_uri, registered_name


@pytest.mark.parametrize(("quantile", "expected"), [(0.5, "q50"), (0.9, "q90"), (0.05, "q5")])
def test_head_for_names_the_quantile(quantile, expected):
    assert head_for(quantile) == expected


def test_the_two_heads_never_share_a_name():
    base = "shelfcast-demand-forecaster"
    names = {registered_name(base, head) for head in HEADS.values()}
    assert len(names) == len(HEADS)


def test_head_uri_addresses_a_stage_of_one_head():
    assert (
        head_uri("shelfcast-demand-forecaster", "q90", "Production")
        == "models:/shelfcast-demand-forecaster-q90/Production"
    )


def test_training_artifact_path_matches_the_registered_head():
    # registry.register reads runs:/<id>/<artifact_path>; train.py writes it.
    assert artifact_path(head_for(0.5)) == "model_q50"


def test_heads_cover_the_labels_serving_returns():
    assert set(HEADS) == {"p50", "p90"}
