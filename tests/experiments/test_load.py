"""Tests for load_experiment() in src/experiments/results.py."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.experiments.results import ExperimentResult, load_experiment, save_experiment

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def saved_result(tmp_path: Path) -> tuple[ExperimentResult, Path]:
    idx = pd.date_range("2021-01-01", periods=30, freq="B")
    rng = np.random.default_rng(5)
    net_ret = pd.Series(rng.normal(0.0003, 0.01, 30), index=idx)
    equity = (1.0 + net_ret).cumprod()
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)

    result = ExperimentResult(
        experiment_name="roundtrip_test",
        strategy_name="TestStrategy(p=1)",
        parameters={"p": 1, "freq": "ME"},
        metrics={"annualized_return": 0.10, "sharpe_ratio": 0.75,
                 "max_drawdown": -0.05, "annualized_volatility": 0.12,
                 "calmar_ratio": 2.0, "hit_rate": 0.55},
        weights=weights,
        equity_curve=equity,
        returns=net_ret,
        created_at=datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC),
    )
    out_path = save_experiment(result, output_dir=tmp_path)
    return result, out_path


# ---------------------------------------------------------------------------
# load_experiment — roundtrip correctness
# ---------------------------------------------------------------------------


def test_load_returns_experiment_result(saved_result: tuple) -> None:
    _, path = saved_result
    loaded = load_experiment(path)
    assert isinstance(loaded, ExperimentResult)


def test_load_experiment_name(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    assert loaded.experiment_name == original.experiment_name


def test_load_strategy_name(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    assert loaded.strategy_name == original.strategy_name


def test_load_parameters(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    assert loaded.parameters == original.parameters


def test_load_metrics(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    for k, v in original.metrics.items():
        assert abs(loaded.metrics[k] - v) < 1e-10


def test_load_equity_curve(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    pd.testing.assert_series_equal(
        loaded.equity_curve.reset_index(drop=True),
        original.equity_curve.reset_index(drop=True),
        check_names=False,
    )


def test_load_returns(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    pd.testing.assert_series_equal(
        loaded.returns.reset_index(drop=True),
        original.returns.reset_index(drop=True),
        check_names=False,
    )


def test_load_weights(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    pd.testing.assert_frame_equal(
        loaded.weights.reset_index(drop=True),
        original.weights.reset_index(drop=True),
        check_freq=False,
    )


def test_load_created_at(saved_result: tuple) -> None:
    original, path = saved_result
    loaded = load_experiment(path)
    assert loaded.created_at == original.created_at


# ---------------------------------------------------------------------------
# load_experiment — error handling
# ---------------------------------------------------------------------------


def test_load_nonexistent_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment(tmp_path / "does_not_exist")
