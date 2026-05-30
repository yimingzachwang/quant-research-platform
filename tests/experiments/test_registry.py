"""Tests for src/experiments/registry.py."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.experiments.config import ExperimentSpec
from src.experiments.registry import (
    ExperimentRegistry,
    latest_experiments,
    load_registry,
    query_registry,
    register_experiment,
)
from src.experiments.results import ExperimentResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(name: str, sharpe: float = 0.6) -> ExperimentResult:
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    rng = np.random.default_rng(0)
    net_ret = pd.Series(rng.normal(0.0003, 0.01, 30), index=idx)
    equity = (1.0 + net_ret).cumprod()
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    return ExperimentResult(
        experiment_name=name,
        strategy_name=f"Strategy_{name}",
        parameters={},
        metrics={"annualized_return": 0.08, "sharpe_ratio": sharpe,
                 "max_drawdown": -0.05, "annualized_volatility": 0.10,
                 "calmar_ratio": 1.6, "hit_rate": 0.52},
        weights=weights,
        equity_curve=equity,
        returns=net_ret,
        created_at=datetime(2026, 5, 22, tzinfo=UTC),
    )


@pytest.fixture()
def registry(tmp_path: Path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "registry.json")


@pytest.fixture()
def populated_registry(tmp_path: Path) -> ExperimentRegistry:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    reg.register(_make_result("exp_a", sharpe=0.8))
    reg.register(_make_result("exp_b", sharpe=0.5))
    reg.register(_make_result("exp_c", sharpe=-0.2))
    return reg


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_returns_string(registry: ExperimentRegistry) -> None:
    exp_id = registry.register(_make_result("test"))
    assert isinstance(exp_id, str)


def test_register_adds_entry(registry: ExperimentRegistry) -> None:
    registry.register(_make_result("test"))
    entries = registry.load()
    assert len(entries) == 1


def test_register_multiple(registry: ExperimentRegistry) -> None:
    registry.register(_make_result("exp_a"))
    registry.register(_make_result("exp_b"))
    assert len(registry.load()) == 2


def test_register_stores_strategy_name(registry: ExperimentRegistry) -> None:
    result = _make_result("test_x")
    registry.register(result)
    entry = registry.load()[0]
    assert entry["strategy_name"] == result.strategy_name


def test_register_stores_metrics_summary(registry: ExperimentRegistry) -> None:
    registry.register(_make_result("test"))
    entry = registry.load()[0]
    assert "sharpe_ratio" in entry["metrics_summary"]


def test_register_idempotent_overwrites(registry: ExperimentRegistry) -> None:
    r1 = _make_result("same_name", sharpe=0.6)
    r2 = _make_result("same_name", sharpe=0.9)
    registry.register(r1)
    registry.register(r2)
    entries = registry.load()
    assert len(entries) == 1
    assert entries[0]["metrics_summary"]["sharpe_ratio"] == pytest.approx(0.9)


def test_register_with_spec_stores_tags(registry: ExperimentRegistry) -> None:
    result = _make_result("tagged")
    spec = ExperimentSpec(
        experiment_name="tagged",
        strategy_name="S",
        universe=["A"],
        start_date="2020-01-01",
        end_date="2021-01-01",
        rebalance_frequency="ME",
        parameters={},
        tags=["momentum", "etf"],
    )
    registry.register(result, spec=spec)
    entry = registry.load()[0]
    assert "momentum" in entry["tags"]
    assert "etf" in entry["tags"]


def test_register_with_spec_stores_config_hash(registry: ExperimentRegistry) -> None:
    from src.experiments.config import experiment_hash
    result = _make_result("hashed")
    spec = ExperimentSpec(
        experiment_name="hashed",
        strategy_name="S",
        universe=["A"],
        start_date="2020-01-01",
        end_date="2021-01-01",
        rebalance_frequency="ME",
        parameters={"k": 5},
    )
    registry.register(result, spec=spec)
    entry = registry.load()[0]
    assert entry["config_hash"] == experiment_hash(spec)


# ---------------------------------------------------------------------------
# load / latest
# ---------------------------------------------------------------------------


def test_load_empty_registry_returns_empty_list(registry: ExperimentRegistry) -> None:
    assert registry.load() == []


def test_load_returns_list(populated_registry: ExperimentRegistry) -> None:
    entries = populated_registry.load()
    assert isinstance(entries, list)


def test_latest_returns_n_entries(populated_registry: ExperimentRegistry) -> None:
    latest = populated_registry.latest(n=2)
    assert len(latest) == 2


def test_latest_default_n(populated_registry: ExperimentRegistry) -> None:
    latest = populated_registry.latest()
    assert len(latest) <= 10


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def test_query_by_strategy_name(populated_registry: ExperimentRegistry) -> None:
    results = populated_registry.query(strategy_name="Strategy_exp_a")
    assert len(results) == 1
    assert results[0]["experiment_name"] == "exp_a"


def test_query_by_strategy_name_substring(populated_registry: ExperimentRegistry) -> None:
    results = populated_registry.query(strategy_name="Strategy_exp")
    assert len(results) == 3


def test_query_by_min_sharpe(populated_registry: ExperimentRegistry) -> None:
    results = populated_registry.query(min_sharpe=0.6)
    names = [e["experiment_name"] for e in results]
    assert "exp_a" in names
    assert "exp_c" not in names


def test_query_by_tags(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    spec1 = ExperimentSpec(
        experiment_name="t1", strategy_name="S", universe=["A"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={}, tags=["momentum"],
    )
    spec2 = ExperimentSpec(
        experiment_name="t2", strategy_name="S", universe=["A"],
        start_date="2020-01-01", end_date="2021-01-01",
        rebalance_frequency="ME", parameters={}, tags=["baseline"],
    )
    reg.register(_make_result("t1"), spec=spec1)
    reg.register(_make_result("t2"), spec=spec2)
    results = reg.query(tags=["momentum"])
    assert len(results) == 1
    assert results[0]["experiment_name"] == "t1"


def test_query_no_filters_returns_all(populated_registry: ExperimentRegistry) -> None:
    assert len(populated_registry.query()) == 3


# ---------------------------------------------------------------------------
# get / remove
# ---------------------------------------------------------------------------


def test_get_existing(populated_registry: ExperimentRegistry) -> None:
    entry = populated_registry.get("exp_a")
    assert entry is not None
    assert entry["experiment_name"] == "exp_a"


def test_get_nonexistent_returns_none(populated_registry: ExperimentRegistry) -> None:
    assert populated_registry.get("does_not_exist") is None


def test_remove_existing(populated_registry: ExperimentRegistry) -> None:
    removed = populated_registry.remove("exp_b")
    assert removed is True
    assert populated_registry.get("exp_b") is None
    assert len(populated_registry.load()) == 2


def test_remove_nonexistent_returns_false(populated_registry: ExperimentRegistry) -> None:
    assert populated_registry.remove("ghost") is False


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def test_register_experiment_function(tmp_path: Path) -> None:
    exp_id = register_experiment(
        _make_result("func_test"),
        registry_path=tmp_path / "registry.json",
    )
    assert isinstance(exp_id, str)


def test_load_registry_function(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "r.json")
    reg.register(_make_result("x"))
    entries = load_registry(tmp_path / "r.json")
    assert len(entries) == 1


def test_query_registry_function(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "r.json")
    reg.register(_make_result("high", sharpe=1.2))
    reg.register(_make_result("low", sharpe=0.1))
    results = query_registry(min_sharpe=1.0, registry_path=tmp_path / "r.json")
    assert len(results) == 1


def test_latest_experiments_function(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "r.json")
    for i in range(5):
        reg.register(_make_result(f"exp_{i}"))
    latest = latest_experiments(n=3, registry_path=tmp_path / "r.json")
    assert len(latest) == 3
