"""Tests for ExperimentSpec, experiment_hash, to_dict, and save_config."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.config import ExperimentSpec, experiment_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_name="test_momentum",
        strategy_name="MomentumRotation(lookback=252)",
        universe=["SPY", "QQQ", "IWM", "TLT", "GLD"],
        start_date="2015-01-01",
        end_date="2024-12-31",
        rebalance_frequency="ME",
        parameters={"lookback": 252, "top_n": 3},
        tags=["momentum", "etf"],
        description="Test experiment",
    )


# ---------------------------------------------------------------------------
# ExperimentSpec construction
# ---------------------------------------------------------------------------


def test_spec_construction(spec: ExperimentSpec) -> None:
    assert spec.experiment_name == "test_momentum"
    assert spec.strategy_name == "MomentumRotation(lookback=252)"
    assert "SPY" in spec.universe
    assert spec.rebalance_frequency == "ME"


def test_spec_tags_default_to_empty_list() -> None:
    s = ExperimentSpec(
        experiment_name="x",
        strategy_name="S",
        universe=["A"],
        start_date="2020-01-01",
        end_date="2021-01-01",
        rebalance_frequency="ME",
        parameters={},
    )
    assert s.tags == []
    assert s.description == ""


# ---------------------------------------------------------------------------
# to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict(spec: ExperimentSpec) -> None:
    d = spec.to_dict()
    assert isinstance(d, dict)


def test_to_dict_contains_all_fields(spec: ExperimentSpec) -> None:
    d = spec.to_dict()
    for key in ["experiment_name", "strategy_name", "universe", "start_date",
                "end_date", "rebalance_frequency", "parameters", "tags", "description"]:
        assert key in d


def test_from_dict_roundtrip(spec: ExperimentSpec) -> None:
    d = spec.to_dict()
    restored = ExperimentSpec.from_dict(d)
    assert restored.experiment_name == spec.experiment_name
    assert restored.strategy_name == spec.strategy_name
    assert restored.universe == spec.universe
    assert restored.parameters == spec.parameters
    assert restored.tags == spec.tags


def test_to_dict_json_serializable(spec: ExperimentSpec) -> None:
    d = spec.to_dict()
    serialized = json.dumps(d)  # must not raise
    assert isinstance(serialized, str)


def test_to_dict_with_numpy_parameters() -> None:
    s = ExperimentSpec(
        experiment_name="np_test",
        strategy_name="S",
        universe=["A"],
        start_date="2020-01-01",
        end_date="2021-01-01",
        rebalance_frequency="ME",
        parameters={"n": np.int64(3), "scale": np.float64(0.5)},
    )
    d = s.to_dict()
    serialized = json.dumps(d)  # must not raise with numpy types
    reloaded = json.loads(serialized)
    assert reloaded["parameters"]["n"] == 3
    assert reloaded["parameters"]["scale"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# save_config / load_config roundtrip
# ---------------------------------------------------------------------------


def test_save_config_creates_file(spec: ExperimentSpec, tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    spec.save_config(out)
    assert out.exists()


def test_save_config_is_valid_json(spec: ExperimentSpec, tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    spec.save_config(out)
    with out.open() as f:
        data = json.load(f)
    assert data["experiment_name"] == spec.experiment_name


def test_load_config_roundtrip(spec: ExperimentSpec, tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    spec.save_config(out)
    loaded = ExperimentSpec.load_config(out)
    assert loaded.experiment_name == spec.experiment_name
    assert loaded.strategy_name == spec.strategy_name
    assert loaded.universe == spec.universe
    assert loaded.parameters == spec.parameters
    assert loaded.tags == spec.tags


# ---------------------------------------------------------------------------
# experiment_hash
# ---------------------------------------------------------------------------


def test_experiment_hash_returns_string(spec: ExperimentSpec) -> None:
    h = experiment_hash(spec)
    assert isinstance(h, str)


def test_experiment_hash_length_12(spec: ExperimentSpec) -> None:
    h = experiment_hash(spec)
    assert len(h) == 12


def test_experiment_hash_deterministic(spec: ExperimentSpec) -> None:
    h1 = experiment_hash(spec)
    h2 = experiment_hash(spec)
    assert h1 == h2


def test_experiment_hash_identical_specs_equal() -> None:
    s1 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={"k": 5},
    )
    s2 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={"k": 5},
    )
    assert experiment_hash(s1) == experiment_hash(s2)


def test_experiment_hash_different_parameters_differ() -> None:
    s1 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={"k": 5},
    )
    s2 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={"k": 10},
    )
    assert experiment_hash(s1) != experiment_hash(s2)


def test_experiment_hash_tags_do_not_affect_hash() -> None:
    s1 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={}, tags=["tag1"],
    )
    s2 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={}, tags=["tag2", "tag3"],
    )
    assert experiment_hash(s1) == experiment_hash(s2)


def test_experiment_hash_description_does_not_affect_hash() -> None:
    s1 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={}, description="old",
    )
    s2 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={}, description="new",
    )
    assert experiment_hash(s1) == experiment_hash(s2)


def test_experiment_hash_universe_order_independent() -> None:
    s1 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X", "Y"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={},
    )
    s2 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["Y", "X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={},
    )
    assert experiment_hash(s1) == experiment_hash(s2)


def test_experiment_hash_different_dates_differ() -> None:
    s1 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={},
    )
    s2 = ExperimentSpec(
        experiment_name="A", strategy_name="S", universe=["X"],
        start_date="2019-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={},
    )
    assert experiment_hash(s1) != experiment_hash(s2)
