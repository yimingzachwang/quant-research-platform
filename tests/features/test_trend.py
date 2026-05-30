"""Tests for src/features/trend.py."""

import pandas as pd
import pytest
from src.features.trend import ema, sma, sma_crossover, trend_strength


@pytest.fixture()
def prices_50() -> pd.Series:
    return pd.Series([100.0 + i for i in range(50)], name="close")


def test_sma_nan_before_window(prices_50: pd.Series) -> None:
    result = sma(prices_50, 10)
    assert result.iloc[:9].isna().all()
    assert result.iloc[9:].notna().all()


def test_sma_value(prices_50: pd.Series) -> None:
    result = sma(prices_50, 5)
    # bar 4: mean of [100, 101, 102, 103, 104] = 102
    assert result.iloc[4] == pytest.approx(102.0)


def test_sma_name(prices_50: pd.Series) -> None:
    assert sma(prices_50, 20).name == "sma_20"


def test_ema_no_leading_nan(prices_50: pd.Series) -> None:
    result = ema(prices_50, span=10)
    assert result.notna().all()


def test_ema_name(prices_50: pd.Series) -> None:
    assert ema(prices_50, 20).name == "ema_20"


def test_sma_crossover_uptrend(prices_50: pd.Series) -> None:
    result = sma_crossover(prices_50, fast=5, slow=10)
    # In a rising linear series fast SMA > slow SMA after warm-up
    assert (result.iloc[10:] == 1).all()


def test_sma_crossover_name(prices_50: pd.Series) -> None:
    assert sma_crossover(prices_50, 5, 20).name == "sma_cross_5_20"


def test_trend_strength_perfect_uptrend(prices_50: pd.Series) -> None:
    result = trend_strength(prices_50, window=10)
    # Linear uptrend → correlation = 1.0
    assert result.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_trend_strength_perfect_downtrend() -> None:
    prices = pd.Series([50.0 - i for i in range(30)])
    result = trend_strength(prices, window=10)
    assert result.iloc[-1] == pytest.approx(-1.0, abs=1e-6)


def test_trend_strength_range(prices_50: pd.Series) -> None:
    result = trend_strength(prices_50, window=10)
    valid = result.dropna()
    assert (valid >= -1).all() and (valid <= 1).all()
