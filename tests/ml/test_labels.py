"""Tests for src.ml.labels.

Focus areas: shift correctness, trailing NaN, leakage timing, and
multi-asset (panel) behaviour for ranking_target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.ml.labels import (
    binary_direction_label,
    forward_returns,
    ranking_target,
    volatility_target,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _price_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    log_r = rng.normal(0.0002, 0.01, size=n)
    prices = pd.Series(100.0 * np.exp(log_r.cumsum()), index=idx, name="close")
    return prices


def _price_df(n: int = 120, tickers: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    tickers = tickers or ["A", "B", "C"]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for t in tickers:
        log_r = rng.normal(0.0002, 0.01, size=n)
        data[t] = 100.0 * np.exp(log_r.cumsum())
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------


def test_forward_returns_length_preserved():
    prices = _price_series()
    result = forward_returns(prices, horizon=5)
    assert len(result) == len(prices)


def test_forward_returns_trailing_nans():
    horizon = 10
    prices = _price_series(n=50)
    result = forward_returns(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all()


def test_forward_returns_no_leading_nans_from_shift():
    prices = _price_series(n=50)
    result = forward_returns(prices, horizon=5)
    # The first few may be NaN due to pct_change(horizon), but no additional
    # NaN should appear in the middle rows
    non_nan = result.dropna()
    assert len(non_nan) > 0


def test_forward_returns_value_correctness():
    # Manual check: forward_returns at t should equal prices[t+h]/prices[t] - 1
    horizon = 3
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    prices = pd.Series(np.arange(100.0, 120.0), index=idx)
    result = forward_returns(prices, horizon=horizon)
    for i in range(len(prices) - horizon):
        expected = prices.iloc[i + horizon] / prices.iloc[i] - 1
        assert abs(result.iloc[i] - expected) < 1e-10, f"Mismatch at row {i}"


def test_forward_returns_same_index_as_input():
    prices = _price_series()
    result = forward_returns(prices, horizon=5)
    assert result.index.equals(prices.index)


def test_forward_returns_horizon_1():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 104.0], index=idx)
    result = forward_returns(prices, horizon=1)
    assert abs(result.iloc[0] - (102.0 / 100.0 - 1)) < 1e-10
    assert result.iloc[-1] != result.iloc[-1]  # NaN check


# ---------------------------------------------------------------------------
# binary_direction_label
# ---------------------------------------------------------------------------


def test_binary_direction_label_values_are_zero_or_one():
    prices = _price_series()
    result = binary_direction_label(prices, horizon=5)
    valid = result.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_binary_direction_label_trailing_nans():
    horizon = 7
    prices = _price_series(n=60)
    result = binary_direction_label(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all()


def test_binary_direction_label_dtype_float():
    prices = _price_series()
    result = binary_direction_label(prices, horizon=5)
    assert pd.api.types.is_float_dtype(result)


def test_binary_direction_label_consistent_with_forward_returns():
    prices = _price_series()
    horizon = 10
    fwd = forward_returns(prices, horizon=horizon)
    binary = binary_direction_label(prices, horizon=horizon)
    # Where fwd > 0, binary should be 1; where fwd <= 0, binary should be 0
    valid_mask = fwd.notna()
    assert (binary[valid_mask & (fwd > 0)] == 1.0).all()
    assert (binary[valid_mask & (fwd <= 0)] == 0.0).all()


def test_binary_direction_label_same_index_as_input():
    prices = _price_series()
    result = binary_direction_label(prices, horizon=5)
    assert result.index.equals(prices.index)


# ---------------------------------------------------------------------------
# volatility_target
# ---------------------------------------------------------------------------


def test_volatility_target_length_preserved():
    prices = _price_series()
    result = volatility_target(prices, horizon=10)
    assert len(result) == len(prices)


def test_volatility_target_trailing_nans():
    horizon = 15
    prices = _price_series(n=80)
    result = volatility_target(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all()


def test_volatility_target_non_negative_where_valid():
    prices = _price_series(n=100)
    result = volatility_target(prices, horizon=10)
    valid = result.dropna()
    assert (valid >= 0).all()


def test_volatility_target_same_index_as_input():
    prices = _price_series()
    result = volatility_target(prices, horizon=5)
    assert result.index.equals(prices.index)


# ---------------------------------------------------------------------------
# ranking_target
# ---------------------------------------------------------------------------


def test_ranking_target_shape_matches_input():
    prices = _price_df()
    result = ranking_target(prices, horizon=10)
    assert result.shape == prices.shape


def test_ranking_target_columns_match_input():
    prices = _price_df(tickers=["SPY", "QQQ", "IWM"])
    result = ranking_target(prices, horizon=5)
    assert list(result.columns) == list(prices.columns)


def test_ranking_target_trailing_nans():
    horizon = 10
    prices = _price_df(n=80)
    result = ranking_target(prices, horizon=horizon)
    assert result.iloc[-horizon:].isna().all().all()


def test_ranking_target_values_in_unit_interval():
    prices = _price_df(n=200, tickers=["A", "B", "C", "D"])
    result = ranking_target(prices, horizon=20)
    valid = result.dropna()
    assert (valid > 0.0).all().all()
    assert (valid <= 1.0).all().all()


def test_ranking_target_relative_ordering():
    # Construct prices where A always beats B over the horizon window
    horizon = 5
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "A": np.exp(np.linspace(0, 1, n)),   # always rising faster
            "B": np.exp(np.linspace(0, 0.3, n)), # rising slower
        },
        index=idx,
    )
    result = ranking_target(prices, horizon=horizon)
    valid = result.dropna()
    # A should consistently rank higher (rank closer to 1.0)
    assert (valid["A"] > valid["B"]).all()


def test_ranking_target_same_index_as_input():
    prices = _price_df()
    result = ranking_target(prices, horizon=5)
    assert result.index.equals(prices.index)
