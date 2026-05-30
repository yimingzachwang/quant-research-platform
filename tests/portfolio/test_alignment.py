"""Tests for src/portfolio/alignment.py and panel.py."""

import pandas as pd
import pytest

from src.portfolio.alignment import align_prices, align_returns
from src.portfolio.panel import (
    universe_momentum,
    universe_returns,
    universe_rolling_volatility,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_price_df(n: int = 300, symbols: list[str] | None = None) -> pd.DataFrame:
    """Synthetic Date × Asset price DataFrame with a clean DatetimeIndex."""
    if symbols is None:
        symbols = ["A", "B", "C"]
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {s: 100.0 + (i + 1) * 0.05 * pd.RangeIndex(n) for i, s in enumerate(symbols)}
    return pd.DataFrame(data, index=idx)


def _make_raw_universe(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Dict of raw DataFrames mimicking load_dataset() output (timestamp column, UTC)."""
    if symbols is None:
        symbols = ["A", "B", "C"]
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    result = {}
    for i, sym in enumerate(symbols):
        result[sym] = pd.DataFrame(
            {
                "timestamp": idx,
                "close": 100.0 + i + pd.RangeIndex(100) * 0.1,
            }
        )
    return result


# ---------------------------------------------------------------------------
# align_prices
# ---------------------------------------------------------------------------


def test_align_prices_shape(
) -> None:
    universe = _make_raw_universe(["A", "B", "C"])
    prices = align_prices(universe)
    assert prices.shape[1] == 3
    assert list(prices.columns) == ["A", "B", "C"]


def test_align_prices_datetime_index() -> None:
    universe = _make_raw_universe(["A", "B"])
    prices = align_prices(universe)
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_align_prices_tz_stripped() -> None:
    universe = _make_raw_universe(["A"])
    prices = align_prices(universe)
    assert prices.index.tz is None


def test_align_prices_sorted_ascending() -> None:
    universe = _make_raw_universe(["A", "B"])
    prices = align_prices(universe)
    assert prices.index.is_monotonic_increasing


def test_align_prices_inner_join_no_nan() -> None:
    universe = _make_raw_universe(["A", "B"])
    prices = align_prices(universe, join="inner")
    assert not prices.isna().any().any()


def test_align_prices_invalid_join() -> None:
    universe = _make_raw_universe(["A"])
    with pytest.raises(ValueError, match="join must be"):
        align_prices(universe, join="left")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# align_returns
# ---------------------------------------------------------------------------


def test_align_returns_first_row_nan() -> None:
    prices = _make_price_df(50, ["A", "B"])
    rets = align_returns(prices)
    assert rets.iloc[0].isna().all()


def test_align_returns_shape_preserved() -> None:
    prices = _make_price_df(50, ["A", "B", "C"])
    rets = align_returns(prices)
    assert rets.shape == prices.shape


def test_align_returns_columns_preserved() -> None:
    prices = _make_price_df(50, ["X", "Y"])
    rets = align_returns(prices)
    assert list(rets.columns) == ["X", "Y"]


# ---------------------------------------------------------------------------
# Panel features
# ---------------------------------------------------------------------------


def test_universe_returns_shape() -> None:
    prices = _make_price_df(100, ["A", "B", "C"])
    rets = universe_returns(prices)
    assert rets.shape == prices.shape


def test_universe_momentum_nan_before_window() -> None:
    prices = _make_price_df(300, ["A", "B"])
    mom = universe_momentum(prices, window=252)
    assert mom.iloc[:252].isna().all().all()
    assert mom.iloc[252:].notna().all().all()


def test_universe_momentum_positive_uptrend() -> None:
    prices = _make_price_df(300, ["A"])
    mom = universe_momentum(prices, window=20)
    assert (mom.dropna() > 0).all().all()


def test_universe_rolling_volatility_non_negative() -> None:
    prices = _make_price_df(200, ["A", "B"])
    rets = universe_returns(prices)
    vol = universe_rolling_volatility(rets, window=20)
    valid = vol.dropna()
    assert (valid >= 0).all().all()


def test_universe_rolling_volatility_annualized_larger() -> None:
    prices = _make_price_df(200, ["A"])
    rets = universe_returns(prices)
    raw = universe_rolling_volatility(rets, window=20, annualize=False)
    ann = universe_rolling_volatility(rets, window=20, annualize=True)
    assert (ann.dropna() > raw.dropna()).all().all()
