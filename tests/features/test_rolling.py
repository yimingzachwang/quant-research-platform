"""Tests for src/features/rolling.py."""

import numpy as np
import pandas as pd
import pytest
from src.features.rolling import rolling_minmax, rolling_rank, rolling_zscore


@pytest.fixture()
def series_30() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(100, 10, 30), name="price")


def test_rolling_zscore_nan_before_window(series_30: pd.Series) -> None:
    z = rolling_zscore(series_30, window=10)
    assert z.iloc[:9].isna().all()
    assert z.iloc[9:].notna().all()


def test_rolling_zscore_mean_zero_std_one(series_30: pd.Series) -> None:
    window = 10
    z = rolling_zscore(series_30, window=window)
    # Last window z-scores should be standardized
    last_window = series_30.iloc[-window:]
    expected_z = (last_window.iloc[-1] - last_window.mean()) / last_window.std()
    assert z.iloc[-1] == pytest.approx(expected_z)


def test_rolling_zscore_constant_series() -> None:
    s = pd.Series([5.0] * 20)
    z = rolling_zscore(s, window=10)
    # std is 0 → division produces NaN
    assert z.iloc[9:].isna().all()


def test_rolling_rank_range(series_30: pd.Series) -> None:
    r = rolling_rank(series_30, window=10)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_rolling_rank_monotone() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    r = rolling_rank(s, window=5)
    # last bar (5.0) is the max → rank should be 1.0
    assert r.iloc[-1] == pytest.approx(1.0)


def test_rolling_minmax_range(series_30: pd.Series) -> None:
    mm = rolling_minmax(series_30, window=10)
    valid = mm.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_rolling_minmax_flat_window() -> None:
    s = pd.Series([3.0] * 15)
    mm = rolling_minmax(s, window=10)
    assert mm.iloc[9:].isna().all()


def test_rolling_minmax_extremes() -> None:
    # Window minimum → 0, window maximum → 1
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    mm = rolling_minmax(s, window=5)
    assert mm.iloc[-1] == pytest.approx(1.0)
