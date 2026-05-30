"""Tests for src/backtesting/engine.py — run_backtest()."""

import numpy as np
import pandas as pd
import pytest
from src.backtesting.engine import run_backtest


@pytest.fixture()
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=100, freq="B")


@pytest.fixture()
def flat_returns(dates: pd.DatetimeIndex) -> pd.Series:
    """Constant 1% daily return."""
    return pd.Series(0.01, index=dates, name="returns")


@pytest.fixture()
def long_signal(dates: pd.DatetimeIndex) -> pd.Series:
    """Constant long (+1) signal."""
    return pd.Series(1.0, index=dates, name="signal")


@pytest.fixture()
def zero_signal(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(0.0, index=dates, name="signal")


# --- Output structure ---


def test_run_backtest_returns_dataframe(flat_returns: pd.Series, long_signal: pd.Series) -> None:
    result = run_backtest(flat_returns, long_signal)
    assert isinstance(result, pd.DataFrame)


def test_run_backtest_expected_columns(flat_returns: pd.Series, long_signal: pd.Series) -> None:
    result = run_backtest(flat_returns, long_signal)
    expected = {
        "position", "gross_return", "turnover", "transaction_cost",
        "net_return", "equity_curve", "drawdown",
    }
    assert set(result.columns) == expected


def test_run_backtest_index_matches_returns(flat_returns: pd.Series, long_signal: pd.Series) -> None:
    result = run_backtest(flat_returns, long_signal)
    assert (result.index == flat_returns.index).all()


# --- Look-ahead bias prevention ---


def test_position_is_signal_lagged_by_one(
    flat_returns: pd.Series, long_signal: pd.Series
) -> None:
    # Signal at row t should show up as position at row t+1
    result = run_backtest(flat_returns, long_signal)
    # Row 0 position must be 0 (no prior signal)
    assert result["position"].iloc[0] == 0.0
    # All subsequent positions should be 1.0 (from the constant long signal)
    assert (result["position"].iloc[1:] == 1.0).all()


def test_look_ahead_bias_example() -> None:
    """A signal that turns on at t=5 should first affect returns at t=6."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    returns = pd.Series(0.01, index=idx)
    signal = pd.Series([0.0] * 5 + [1.0] * 5, index=idx)
    result = run_backtest(returns, signal)
    # rows 0-5: position=0 (rows 0-4 flat, row 5 still flat from lag)
    assert (result["position"].iloc[:6] == 0.0).all()
    # rows 6-9: position=1
    assert (result["position"].iloc[6:] == 1.0).all()


# --- Return correctness ---


def test_gross_return_equals_position_times_returns(
    flat_returns: pd.Series, long_signal: pd.Series
) -> None:
    result = run_backtest(flat_returns, long_signal)
    expected = result["position"] * flat_returns
    pd.testing.assert_series_equal(result["gross_return"], expected, check_names=False)


def test_flat_signal_zero_returns(flat_returns: pd.Series, zero_signal: pd.Series) -> None:
    result = run_backtest(flat_returns, zero_signal)
    assert (result["gross_return"] == 0.0).all()
    assert (result["net_return"] == 0.0).all()


def test_zero_returns_flat_equity_curve() -> None:
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    returns = pd.Series(0.0, index=idx)
    signal = pd.Series(1.0, index=idx)
    result = run_backtest(returns, signal)
    assert ((result["equity_curve"] - 1.0).abs() < 1e-12).all()


# --- Transaction costs ---


def test_no_cost_when_zero_bps(flat_returns: pd.Series, long_signal: pd.Series) -> None:
    result = run_backtest(flat_returns, long_signal, transaction_cost_bps=0.0)
    assert (result["transaction_cost"] == 0.0).all()


def test_cost_applied_on_position_change() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    returns = pd.Series(0.0, index=idx)
    # Signal: [0, 1, -1, 1, 0]
    # After shift(1): position = [0, 0, 1, -1, 1]
    # Turnover (abs diff): [0, 0, 1, 2, 2]
    signal = pd.Series([0.0, 1.0, -1.0, 1.0, 0.0], index=idx)
    result = run_backtest(returns, signal, transaction_cost_bps=10.0)
    cost_bps = 10.0 / 10_000
    # Row 0: turnover=0 (fillna), row 1: |0-0|=0
    assert result["transaction_cost"].iloc[1] == pytest.approx(0.0)
    # Row 2: position flips 0→1, turnover=1
    assert result["transaction_cost"].iloc[2] == pytest.approx(1.0 * cost_bps)
    # Row 3: position flips 1→-1, turnover=2
    assert result["transaction_cost"].iloc[3] == pytest.approx(2.0 * cost_bps)


def test_net_return_less_than_gross_when_cost_nonzero() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    returns = pd.Series(0.01, index=idx)
    signal = pd.Series([1.0, -1.0] * 5, index=idx)  # alternating — high turnover
    result = run_backtest(returns, signal, transaction_cost_bps=5.0)
    # Wherever there is turnover, net < gross
    high_turnover = result["turnover"] > 0
    assert (result.loc[high_turnover, "net_return"] < result.loc[high_turnover, "gross_return"]).all()


# --- Equity curve and drawdown ---


def test_equity_curve_starts_at_one() -> None:
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    returns = pd.Series(0.01, index=idx)
    signal = pd.Series(1.0, index=idx)
    result = run_backtest(returns, signal)
    # First row: position=0, net_return=0, equity_curve = (1+0) = 1
    assert result["equity_curve"].iloc[0] == pytest.approx(1.0)


def test_equity_curve_compounds_correctly() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    returns = pd.Series([0.0, 0.10, 0.10], index=idx)
    signal = pd.Series(1.0, index=idx)
    result = run_backtest(returns, signal)
    # position: [0, 1, 1], net_return: [0, 0.10, 0.10]
    # equity: [1.0, 1.10, 1.21]
    assert result["equity_curve"].iloc[0] == pytest.approx(1.0)
    assert result["equity_curve"].iloc[1] == pytest.approx(1.10)
    assert result["equity_curve"].iloc[2] == pytest.approx(1.21)


def test_drawdown_non_positive() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0, 0.01, 50), index=idx)
    signal = pd.Series(1.0, index=idx)
    result = run_backtest(returns, signal)
    assert (result["drawdown"] <= 0.0).all()


def test_drawdown_zero_when_always_at_peak() -> None:
    # Monotonically rising equity → drawdown always 0
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    returns = pd.Series(0.01, index=idx)
    signal = pd.Series(1.0, index=idx)
    result = run_backtest(returns, signal)
    # Row 0 equity=1 (at peak), all others rising → always at peak
    assert (result["drawdown"].abs() < 1e-10).all()


# --- Index alignment ---


def test_partial_index_overlap() -> None:
    idx_r = pd.date_range("2020-01-01", periods=10, freq="B")
    idx_s = pd.date_range("2020-01-05", periods=10, freq="B")
    returns = pd.Series(0.01, index=idx_r)
    signal = pd.Series(1.0, index=idx_s)
    result = run_backtest(returns, signal)
    common = idx_r.intersection(idx_s)
    assert len(result) == len(common)
