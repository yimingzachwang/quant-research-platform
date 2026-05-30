"""Tests for src/experiments/factory.py."""

import pandas as pd
import pytest
from src.experiments.config import ExperimentSpec
from src.experiments.config_io import normalize_config
from src.experiments.factory import (
    UniverseSpec,
    ValidationConfig,
    available_strategies,
    build_experiment_spec,
    build_strategy,
    build_universe_spec,
    build_validation_config,
    build_validation_splits,
)
from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.validation.splits import TimeSplit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_CFG = {
    "name": "test_exp",
    "universe": {"tickers": ["SPY", "QQQ"]},
    "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
    "strategy": {"type": "MomentumRotation"},
}


def _norm(extra: dict | None = None) -> dict:
    cfg = dict(_MINIMAL_CFG)
    if extra:
        cfg.update(extra)
    return normalize_config(cfg)


# ---------------------------------------------------------------------------
# build_strategy
# ---------------------------------------------------------------------------


def test_build_strategy_momentum_rotation() -> None:
    cfg = {"type": "MomentumRotation", "parameters": {}}
    s = build_strategy(cfg)
    assert isinstance(s, MomentumRotationStrategy)


def test_build_strategy_equal_weight() -> None:
    cfg = {"type": "EqualWeight", "parameters": {}}
    s = build_strategy(cfg)
    assert isinstance(s, EqualWeightStrategy)


def test_build_strategy_buy_and_hold() -> None:
    cfg = {"type": "BuyAndHold", "parameters": {}}
    s = build_strategy(cfg)
    assert isinstance(s, BuyAndHoldStrategy)


def test_build_strategy_passes_parameters() -> None:
    cfg = {"type": "MomentumRotation", "parameters": {"lookback": 120, "top_n": 2}}
    s = build_strategy(cfg)
    assert isinstance(s, MomentumRotationStrategy)
    assert s.lookback == 120
    assert s.top_n == 2


def test_build_strategy_unknown_type_raises() -> None:
    cfg = {"type": "NeuralNet", "parameters": {}}
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_strategy(cfg)


def test_build_strategy_no_parameters_key() -> None:
    cfg = {"type": "EqualWeight"}
    s = build_strategy(cfg)
    assert isinstance(s, EqualWeightStrategy)


# ---------------------------------------------------------------------------
# build_universe_spec
# ---------------------------------------------------------------------------


def test_build_universe_spec_returns_dataclass() -> None:
    spec = build_universe_spec(
        {"tickers": ["SPY", "QQQ"]},
        {"start": "2020-01-01", "end": "2023-12-31"},
    )
    assert isinstance(spec, UniverseSpec)


def test_build_universe_spec_tickers_tuple() -> None:
    spec = build_universe_spec(
        {"tickers": ["SPY", "QQQ"]},
        {"start": "2020-01-01", "end": "2023-12-31"},
    )
    assert isinstance(spec.tickers, tuple)
    assert "SPY" in spec.tickers


def test_build_universe_spec_dates() -> None:
    spec = build_universe_spec(
        {"tickers": ["A"]},
        {"start": "2018-01-01", "end": "2022-12-31"},
    )
    assert spec.start_date == "2018-01-01"
    assert spec.end_date == "2022-12-31"


def test_build_universe_spec_is_frozen() -> None:
    spec = build_universe_spec({"tickers": ["A"]}, {"start": "2020-01-01", "end": "2021-01-01"})
    with pytest.raises((AttributeError, TypeError)):
        spec.tickers = ("B",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_validation_config
# ---------------------------------------------------------------------------


def test_build_validation_config_none_type() -> None:
    cfg = {"type": "none", "parameters": {}}
    vc = build_validation_config(cfg)
    assert vc.type == "none"


def test_build_validation_config_rolling() -> None:
    cfg = {"type": "rolling", "parameters": {"train_months": 36, "test_months": 12}}
    vc = build_validation_config(cfg)
    assert vc.type == "rolling"
    assert vc.parameters["train_months"] == 36


def test_build_validation_config_returns_dataclass() -> None:
    vc = build_validation_config({"type": "none", "parameters": {}})
    assert isinstance(vc, ValidationConfig)


def test_build_validation_config_parameters_is_dict() -> None:
    vc = build_validation_config({"type": "none", "parameters": {"k": 1}})
    assert isinstance(vc.parameters, dict)


# ---------------------------------------------------------------------------
# build_experiment_spec
# ---------------------------------------------------------------------------


def test_build_experiment_spec_returns_spec() -> None:
    norm = _norm()
    spec = build_experiment_spec(norm)
    assert isinstance(spec, ExperimentSpec)


def test_build_experiment_spec_name() -> None:
    norm = _norm()
    spec = build_experiment_spec(norm)
    assert spec.experiment_name == "test_exp"


def test_build_experiment_spec_universe_list() -> None:
    norm = _norm()
    spec = build_experiment_spec(norm)
    assert isinstance(spec.universe, list)
    assert "SPY" in spec.universe


def test_build_experiment_spec_tags() -> None:
    cfg = {**_MINIMAL_CFG, "tags": ["momentum", "etf"]}
    norm = normalize_config(cfg)
    spec = build_experiment_spec(norm)
    assert "momentum" in spec.tags


def test_build_experiment_spec_description() -> None:
    cfg = {**_MINIMAL_CFG, "description": "A test experiment."}
    norm = normalize_config(cfg)
    spec = build_experiment_spec(norm)
    assert spec.description == "A test experiment."


# ---------------------------------------------------------------------------
# build_validation_splits
# ---------------------------------------------------------------------------


def _make_index(n_years: int = 6) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n_years * 252, freq="B")


def test_build_validation_splits_none_returns_empty() -> None:
    vc = ValidationConfig(type="none", parameters={})
    splits = build_validation_splits(vc, _make_index())
    assert splits == []


def test_build_validation_splits_rolling_returns_list() -> None:
    vc = ValidationConfig(
        type="rolling",
        parameters={"train_months": 24, "test_months": 12},
    )
    splits = build_validation_splits(vc, _make_index())
    assert isinstance(splits, list)
    assert len(splits) > 0


def test_build_validation_splits_rolling_elements_are_time_splits() -> None:
    vc = ValidationConfig(
        type="rolling",
        parameters={"train_months": 24, "test_months": 12},
    )
    splits = build_validation_splits(vc, _make_index())
    assert all(isinstance(s, TimeSplit) for s in splits)


def test_build_validation_splits_expanding_returns_list() -> None:
    vc = ValidationConfig(
        type="expanding",
        parameters={"train_months": 24, "test_months": 12},
    )
    splits = build_validation_splits(vc, _make_index())
    assert len(splits) > 0


def test_build_validation_splits_missing_params_raises() -> None:
    vc = ValidationConfig(type="rolling", parameters={})
    with pytest.raises(ValueError, match="train_months"):
        build_validation_splits(vc, _make_index())


def test_build_validation_splits_unknown_type_raises() -> None:
    vc = ValidationConfig(type="walk_forward", parameters={})
    with pytest.raises(ValueError, match="Unknown validation"):
        build_validation_splits(vc, _make_index())


# ---------------------------------------------------------------------------
# available_strategies
# ---------------------------------------------------------------------------


def test_available_strategies_returns_list() -> None:
    result = available_strategies()
    assert isinstance(result, list)


def test_available_strategies_contains_known_types() -> None:
    result = available_strategies()
    assert "MomentumRotation" in result
    assert "EqualWeight" in result
    assert "BuyAndHold" in result


def test_available_strategies_sorted() -> None:
    result = available_strategies()
    assert result == sorted(result)
