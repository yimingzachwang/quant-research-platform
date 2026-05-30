"""Tests for src/features/momentum.py."""

import pandas as pd
import pytest
from src.features.momentum import momentum, momentum_20d, momentum_60d, momentum_252d


@pytest.fixture()
def prices_100() -> pd.Series:
    """100-bar price series growing linearly from 100."""
    return pd.Series([100.0 + i for i in range(100)], name="close")


def test_momentum_nan_for_early_bars(prices_100: pd.Series) -> None:
    result = momentum(prices_100, 20)
    assert result.iloc[:20].isna().all()


def test_momentum_value_correctness(prices_100: pd.Series) -> None:
    result = momentum(prices_100, 20)
    # bar 20: price = 120, shifted = 100  → 20/100 = 0.2
    assert result.iloc[20] == pytest.approx(20 / 100)


def test_momentum_name(prices_100: pd.Series) -> None:
    assert momentum(prices_100, 20).name == "momentum_20d"
    assert momentum_20d(prices_100).name == "momentum_20d"
    assert momentum_60d(prices_100).name == "momentum_60d"


def test_momentum_252d_requires_252_bars() -> None:
    short = pd.Series([100.0 + i for i in range(50)])
    result = momentum_252d(short)
    assert result.notna().sum() == 0


def test_momentum_negative() -> None:
    prices = pd.Series([110.0, 100.0])
    result = momentum(prices, 1)
    assert result.iloc[1] == pytest.approx(-10 / 110)
