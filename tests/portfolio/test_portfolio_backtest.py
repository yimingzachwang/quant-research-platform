"""Tests for src/portfolio/portfolio_backtest.py."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.portfolio_backtest import PortfolioBacktestResult, run_portfolio_backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=100, freq="B")


@pytest.fixture()
def flat_returns(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"A": 0.01, "B": 0.005}, index=dates)


@pytest.fixture()
def equal_weights(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)


@pytest.fixture()
def zero_weights(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_returns_result_type(flat_returns: pd.DataFrame, equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights)
    assert isinstance(result, PortfolioBacktestResult)


def test_backtest_columns(flat_returns: pd.DataFrame, equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights)
    expected = {"gross_return", "turnover", "transaction_cost", "net_return",
                "equity_curve", "drawdown"}
    assert set(result.backtest.columns) == expected


def test_metrics_keys(flat_returns: pd.DataFrame, equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights)
    expected = {"annualized_return", "annualized_volatility", "sharpe_ratio",
                "max_drawdown", "calmar_ratio", "hit_rate"}
    assert set(result.metrics.keys()) == expected


# ---------------------------------------------------------------------------
# Look-ahead prevention
# ---------------------------------------------------------------------------


def test_first_row_position_is_zero(flat_returns: pd.DataFrame, equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights)
    # First row: lagged weights from before any signal = 0 → gross_return = 0
    assert result.backtest["gross_return"].iloc[0] == pytest.approx(0.0)


def test_weights_lagged_by_one(dates: pd.DatetimeIndex) -> None:
    """Signal turns on at row 5 → should affect returns from row 6 onward."""
    returns = pd.DataFrame({"A": 0.01}, index=dates)
    weights = pd.DataFrame({"A": [0.0] * 5 + [1.0] * 95}, index=dates)
    result = run_portfolio_backtest(returns, weights)
    assert (result.backtest["gross_return"].iloc[:6].abs() < 1e-12).all()
    assert ((result.backtest["gross_return"].iloc[6:] - 0.01).abs() < 1e-12).all()


# ---------------------------------------------------------------------------
# Return correctness
# ---------------------------------------------------------------------------


def test_gross_return_is_weighted_sum(flat_returns: pd.DataFrame,
                                      equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights)
    # After first row: gross = 0.5*0.01 + 0.5*0.005 = 0.0075
    assert ((result.backtest["gross_return"].iloc[1:] - 0.0075).abs() < 1e-12).all()


def test_zero_weights_zero_returns(flat_returns: pd.DataFrame,
                                   zero_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, zero_weights)
    assert (result.backtest["net_return"] == 0.0).all()


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


def test_no_cost_when_zero_bps(flat_returns: pd.DataFrame,
                               equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights, transaction_cost_bps=0.0)
    assert (result.backtest["transaction_cost"] == 0.0).all()


def test_cost_reduces_net_return(dates: pd.DatetimeIndex) -> None:
    returns = pd.DataFrame({"A": 0.01, "B": 0.01}, index=dates)
    # Alternating weights → high turnover
    w = [{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}] * 50
    weights = pd.DataFrame(w, index=dates)
    no_cost = run_portfolio_backtest(returns, weights, transaction_cost_bps=0.0)
    with_cost = run_portfolio_backtest(returns, weights, transaction_cost_bps=10.0)
    # Net return should be lower with costs
    assert with_cost.backtest["net_return"].sum() < no_cost.backtest["net_return"].sum()


# ---------------------------------------------------------------------------
# Equity curve & drawdown
# ---------------------------------------------------------------------------


def test_equity_curve_starts_at_one(flat_returns: pd.DataFrame,
                                    equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights)
    assert result.backtest["equity_curve"].iloc[0] == pytest.approx(1.0)


def test_drawdown_non_positive(flat_returns: pd.DataFrame,
                               equal_weights: pd.DataFrame) -> None:
    result = run_portfolio_backtest(flat_returns, equal_weights)
    assert (result.backtest["drawdown"] <= 0.0).all()


def test_drawdown_zero_for_monotone_rising() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    returns = pd.DataFrame({"A": 0.01}, index=idx)
    weights = pd.DataFrame({"A": 1.0}, index=idx)
    result = run_portfolio_backtest(returns, weights)
    assert (result.backtest["drawdown"].abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# Index alignment
# ---------------------------------------------------------------------------


def test_partial_index_overlap() -> None:
    idx_r = pd.date_range("2020-01-01", periods=20, freq="B")
    idx_w = pd.date_range("2020-01-10", periods=20, freq="B")
    returns = pd.DataFrame({"A": 0.01}, index=idx_r)
    weights = pd.DataFrame({"A": 1.0}, index=idx_w)
    result = run_portfolio_backtest(returns, weights)
    common = idx_r.intersection(idx_w)
    assert len(result.backtest) == len(common)
