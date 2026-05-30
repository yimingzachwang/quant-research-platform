"""Tests for src/backtesting/signals.py."""

import numpy as np
import pandas as pd
import pytest
from src.backtesting.signals import (
    crossover_signal,
    long_only_signal,
    signal_from_threshold,
    volatility_target_signal,
)


@pytest.fixture()
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=50, freq="B")


def test_long_only_true_condition(dates: pd.DatetimeIndex) -> None:
    condition = pd.Series(True, index=dates)
    s = long_only_signal(condition)
    assert (s == 1.0).all()


def test_long_only_false_condition(dates: pd.DatetimeIndex) -> None:
    condition = pd.Series(False, index=dates)
    s = long_only_signal(condition)
    assert (s == 0.0).all()


def test_long_only_no_short(dates: pd.DatetimeIndex) -> None:
    # Boolean condition should never produce negative values
    condition = pd.Series([True, False, True] * 16 + [True, False], index=dates)
    s = long_only_signal(condition)
    assert (s >= 0).all()
    assert s.isin([0.0, 1.0]).all()


def test_signal_from_threshold_above(dates: pd.DatetimeIndex) -> None:
    values = pd.Series(range(50), index=dates, dtype=float)
    s = signal_from_threshold(values, threshold=25.0, direction="above")
    assert s.iloc[:26].sum() == 0.0
    assert (s.iloc[26:] == 1.0).all()


def test_signal_from_threshold_below(dates: pd.DatetimeIndex) -> None:
    values = pd.Series(range(50), index=dates, dtype=float)
    s = signal_from_threshold(values, threshold=25.0, direction="below")
    assert (s.iloc[:25] == 1.0).all()
    assert s.iloc[25:].sum() == 0.0


def test_signal_from_threshold_invalid_direction(dates: pd.DatetimeIndex) -> None:
    values = pd.Series(range(10), index=dates[:10], dtype=float)
    with pytest.raises(ValueError, match="direction must be"):
        signal_from_threshold(values, threshold=5.0, direction="sideways")  # type: ignore[arg-type]


def test_crossover_signal_fast_above(dates: pd.DatetimeIndex) -> None:
    fast = pd.Series(2.0, index=dates)
    slow = pd.Series(1.0, index=dates)
    s = crossover_signal(fast, slow)
    assert (s == 1.0).all()


def test_crossover_signal_fast_below(dates: pd.DatetimeIndex) -> None:
    fast = pd.Series(1.0, index=dates)
    slow = pd.Series(2.0, index=dates)
    s = crossover_signal(fast, slow)
    assert (s == -1.0).all()


def test_crossover_signal_equal(dates: pd.DatetimeIndex) -> None:
    fast = pd.Series(1.0, index=dates)
    slow = pd.Series(1.0, index=dates)
    s = crossover_signal(fast, slow)
    assert (s == 0.0).all()


def test_crossover_signal_values_in_set(dates: pd.DatetimeIndex) -> None:
    rng = np.random.default_rng(1)
    fast = pd.Series(rng.normal(0, 1, 50), index=dates)
    slow = pd.Series(rng.normal(0, 1, 50), index=dates)
    s = crossover_signal(fast, slow)
    assert s.isin([-1.0, 0.0, 1.0]).all()


def test_volatility_target_signal_scales_down(dates: pd.DatetimeIndex) -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.05, 50), index=dates)
    signal = pd.Series(1.0, index=dates)
    result = volatility_target_signal(signal, returns, target_vol=0.10, window=20)
    valid = result.dropna()
    # Realized vol ≈ 0.05 * sqrt(252) ≈ 0.79 → weight ≈ 0.10/0.79 ≈ 0.13
    # All values should be well below 1.0
    assert (valid < 1.0).all()


def test_volatility_target_signal_max_leverage_cap(dates: pd.DatetimeIndex) -> None:
    # Very low realized vol → weight would be huge without cap
    returns = pd.Series(0.0001, index=dates)
    signal = pd.Series(1.0, index=dates)
    result = volatility_target_signal(
        signal, returns, target_vol=0.20, window=10, max_leverage=1.0
    )
    valid = result.dropna()
    assert (valid <= 1.0).all()


def test_volatility_target_signal_zero_realized_vol_is_nan(
    dates: pd.DatetimeIndex,
) -> None:
    # Constant returns → std = 0 → realized_vol = 0 → weight = NaN
    returns = pd.Series(0.01, index=dates)
    signal = pd.Series(1.0, index=dates)
    result = volatility_target_signal(signal, returns, target_vol=0.10, window=20)
    assert result.dropna().empty
