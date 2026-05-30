"""Tests for src/strategies/runner.py."""

import numpy as np
import pandas as pd
import pytest
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.strategies.runner import StrategyResult, run_strategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300)),
            "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, 300)),
            "C": 100 * np.cumprod(1 + rng.normal(-0.001, 0.01, 300)),
        },
        index=idx,
    )


@pytest.fixture()
def strategy() -> MomentumRotationStrategy:
    return MomentumRotationStrategy(lookback=60, top_n=2, rebalance_freq="ME")


# ---------------------------------------------------------------------------
# Output type and structure
# ---------------------------------------------------------------------------


def test_run_strategy_returns_result(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    assert isinstance(result, StrategyResult)


def test_strategy_name_in_result(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    assert result.strategy_name == strategy.name


def test_backtest_columns(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    expected = {"gross_return", "turnover", "transaction_cost", "net_return",
                "equity_curve", "drawdown"}
    assert set(result.backtest.columns) == expected


def test_metrics_keys(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    expected = {"annualized_return", "annualized_volatility", "sharpe_ratio",
                "max_drawdown", "calmar_ratio", "hit_rate"}
    assert set(result.metrics.keys()) == expected


def test_weights_shape(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    # Weights columns match price columns
    assert list(result.weights.columns) == list(prices.columns)


# ---------------------------------------------------------------------------
# Look-ahead prevention
# ---------------------------------------------------------------------------


def test_first_row_zero_return(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    assert result.backtest["net_return"].iloc[0] == pytest.approx(0.0)


def test_equity_curve_starts_at_one(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    assert result.backtest["equity_curve"].iloc[0] == pytest.approx(1.0)


def test_drawdown_non_positive(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy)
    assert (result.backtest["drawdown"] <= 0.0).all()


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


def test_cost_reduces_net_return(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    no_cost = run_strategy(prices, strategy, transaction_cost_bps=0.0)
    with_cost = run_strategy(prices, strategy, transaction_cost_bps=20.0)
    assert with_cost.backtest["net_return"].sum() < no_cost.backtest["net_return"].sum()


def test_zero_cost_transaction_cost_column_is_zero(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    result = run_strategy(prices, strategy, transaction_cost_bps=0.0)
    assert (result.backtest["transaction_cost"] == 0.0).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_run_strategy_is_deterministic(
    prices: pd.DataFrame, strategy: MomentumRotationStrategy
) -> None:
    r1 = run_strategy(prices, strategy)
    r2 = run_strategy(prices, strategy)
    pd.testing.assert_frame_equal(r1.backtest, r2.backtest)
