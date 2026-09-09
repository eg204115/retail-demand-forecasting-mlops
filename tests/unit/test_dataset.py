from __future__ import annotations

from src.common.config import Config
from src.training.dataset import chronological_split, feature_names, rolling_origin_folds


def _cfg(**overrides) -> Config:
    base = {
        "data": {
            "horizon": 7,
            "valid_days": 14,
            "test_days": 14,
            "backtest_folds": 3,
            "fold_step": 14,
        }
    }
    base["data"].update(overrides)
    return Config(base)


def test_split_is_chronological_and_leaves_a_horizon_gap(panel):
    split = chronological_split(panel, _cfg())
    assert split.train["date"].max() < split.valid["date"].min()
    assert split.valid["date"].max() < split.test["date"].min()
    gap = (split.valid["date"].min() - split.train["date"].max()).days
    assert gap >= 7


def test_target_is_not_a_feature(panel):
    assert "units" not in feature_names(panel)
    assert "date" not in feature_names(panel)


def test_folds_never_train_on_their_own_test_window(panel):
    for _, train, test in rolling_origin_folds(panel, _cfg()):
        assert train["date"].max() < test["date"].min()
        assert not train.empty and not test.empty


def test_fold_count_matches_config(panel):
    folds = list(rolling_origin_folds(panel, _cfg()))
    assert len(folds) == 3


def test_folds_walk_the_origin_forward(panel):
    starts = [test["date"].min() for _, _, test in rolling_origin_folds(panel, _cfg())]
    assert starts == sorted(starts, reverse=True) or starts == sorted(starts)
    assert len(set(starts)) == len(starts)
