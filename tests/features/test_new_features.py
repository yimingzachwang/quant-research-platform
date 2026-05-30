"""Tests for new feature functions added in G2.

Covers: downside_volatility, vol_of_vol, vol_percentile (volatility.py)
        bollinger_distance, rolling_skewness, rolling_autocorrelation (rolling.py)
All tests are structural/smoke tests — verify Series return type, name,
non-trivial output, and edge-case safety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.features.rolling import bollinger_distance, rolling_autocorrelation, rolling_skewness
from src.features.volatility import downside_volatility, vol_of_vol, vol_percentile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    data = rng.normal(0.0002, 0.01, n)
    return pd.Series(data, index=idx, name="returns")


def _make_prices(n: int = 300, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    log_ret = rng.normal(0.0002, 0.01, n)
    prices = 100 * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=idx, name="price")


# ---------------------------------------------------------------------------
# downside_volatility
# ---------------------------------------------------------------------------

def test_downside_vol_returns_series():
    ret = _make_returns()
    result = downside_volatility(ret, window=21)
    assert isinstance(result, pd.Series)


def test_downside_vol_name():
    ret = _make_returns()
    result = downside_volatility(ret, window=21)
    assert result.name == "downside_vol_21d"


def test_downside_vol_nan_at_start():
    ret = _make_returns()
    result = downside_volatility(ret, window=21)
    # First window-1 rows have NaN (insufficient observations)
    assert result.iloc[:20].isna().all()


def test_downside_vol_non_negative():
    ret = _make_returns()
    result = downside_volatility(ret, window=21)
    valid = result.dropna()
    assert (valid >= 0).all()


def test_downside_vol_less_than_total_vol():
    ret = _make_returns()
    total = ret.rolling(21).std() * np.sqrt(252)
    downside = downside_volatility(ret, window=21)
    valid_mask = total.notna() & downside.notna()
    # Downside vol should generally be <= total vol (not always exact due to semi-std)
    assert (downside[valid_mask] <= total[valid_mask] * 1.05).all()


def test_downside_vol_no_annualise():
    ret = _make_returns()
    result = downside_volatility(ret, window=21, annualize=False)
    annualised = downside_volatility(ret, window=21, annualize=True)
    valid = result.notna() & annualised.notna()
    # Annualised should be larger by ~sqrt(252)
    ratio = annualised[valid] / result[valid].replace(0, np.nan)
    assert (ratio.dropna().abs() - np.sqrt(252)).abs().mean() < 0.5


def test_downside_vol_single_value_series():
    ret = pd.Series([0.01], index=pd.date_range("2020-01-02", periods=1))
    result = downside_volatility(ret, window=21)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# vol_of_vol
# ---------------------------------------------------------------------------

def test_vol_of_vol_returns_series():
    ret = _make_returns()
    result = vol_of_vol(ret)
    assert isinstance(result, pd.Series)


def test_vol_of_vol_name():
    ret = _make_returns()
    result = vol_of_vol(ret, vol_window=21, meta_window=63)
    assert result.name == "vol_of_vol_21_63"


def test_vol_of_vol_non_negative():
    ret = _make_returns()
    result = vol_of_vol(ret)
    assert (result.dropna() >= 0).all()


def test_vol_of_vol_more_nans_than_inner():
    ret = _make_returns()
    inner = ret.rolling(21).std()
    result = vol_of_vol(ret, vol_window=21, meta_window=63)
    # vol_of_vol requires more history than inner vol alone
    assert result.notna().sum() < inner.notna().sum()


# ---------------------------------------------------------------------------
# vol_percentile
# ---------------------------------------------------------------------------

def test_vol_percentile_returns_series():
    ret = _make_returns()
    result = vol_percentile(ret)
    assert isinstance(result, pd.Series)


def test_vol_percentile_name():
    ret = _make_returns()
    result = vol_percentile(ret, vol_window=21, lookback=252)
    assert result.name == "vol_pct_21_252"


def test_vol_percentile_bounded():
    ret = _make_returns(n=400)
    result = vol_percentile(ret)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_vol_percentile_transition_spike():
    """When vol spikes from low to high, percentile should jump near 1.0.

    When returns shift from low-vol to high-vol, the 21-day rolling vol climbs
    well above the preceding 252-day baseline, so vol_percentile should be close
    to 1.0 in the regime-transition window.
    """
    rng = np.random.default_rng(99)
    idx = pd.date_range("2015-01-02", periods=600, freq="B")
    # Low-vol period (first 300) followed by high-vol period (last 300)
    ret_low = rng.normal(0.0, 0.002, 300)
    ret_high = rng.normal(0.0, 0.04, 300)
    ret = pd.Series(np.concatenate([ret_low, ret_high]), index=idx)
    pct = vol_percentile(ret, vol_window=21, lookback=252)
    # Regime-transition window: rows 310-350 (high-vol just entered,
    # compared against predominantly low-vol lookback → near 1.0 percentile).
    transition_pct = pct.iloc[310:350].dropna()
    assert len(transition_pct) > 0
    assert transition_pct.median() > 0.9


# ---------------------------------------------------------------------------
# bollinger_distance
# ---------------------------------------------------------------------------

def test_bollinger_distance_returns_series():
    prices = _make_prices()
    result = bollinger_distance(prices)
    assert isinstance(result, pd.Series)


def test_bollinger_distance_name():
    prices = _make_prices()
    result = bollinger_distance(prices, window=20)
    assert result.name == "bollinger_20d"


def test_bollinger_distance_nan_at_start():
    prices = _make_prices()
    result = bollinger_distance(prices, window=20)
    assert result.iloc[:19].isna().all()


def test_bollinger_distance_zero_when_at_mean():
    """A constant series should have zero Bollinger distance."""
    idx = pd.date_range("2020-01-02", periods=50, freq="B")
    prices = pd.Series(100.0, index=idx)
    result = bollinger_distance(prices, window=20)
    # std=0 → NaN from replace(0, nan) — this is correct behaviour
    assert result.dropna().empty or (result.dropna().abs() < 1e-10).all()


def test_bollinger_distance_positive_above_mean():
    """A rising series should end with positive distance."""
    idx = pd.date_range("2020-01-02", periods=100, freq="B")
    prices = pd.Series(np.arange(100.0) + 100.0, index=idx)
    result = bollinger_distance(prices, window=20)
    # Last value: price is highest, well above SMA → positive
    assert result.dropna().iloc[-1] > 0


# ---------------------------------------------------------------------------
# rolling_skewness
# ---------------------------------------------------------------------------

def test_rolling_skewness_returns_series():
    ret = _make_returns()
    result = rolling_skewness(ret, window=60)
    assert isinstance(result, pd.Series)


def test_rolling_skewness_name():
    ret = _make_returns()
    result = rolling_skewness(ret, window=60)
    assert result.name == "skew_60d"


def test_rolling_skewness_nan_at_start():
    ret = _make_returns()
    result = rolling_skewness(ret, window=60)
    assert result.iloc[:59].isna().all()


def test_rolling_skewness_symmetric_series():
    """Normal draws should produce near-zero skewness on average."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-02", periods=500, freq="B")
    ret = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    result = rolling_skewness(ret, window=60)
    assert abs(result.dropna().mean()) < 0.5


# ---------------------------------------------------------------------------
# rolling_autocorrelation
# ---------------------------------------------------------------------------

def test_rolling_autocorrelation_returns_series():
    ret = _make_returns()
    result = rolling_autocorrelation(ret, lag=1, window=60)
    assert isinstance(result, pd.Series)


def test_rolling_autocorrelation_name():
    ret = _make_returns()
    result = rolling_autocorrelation(ret, lag=1, window=60)
    assert result.name == "autocorr_1_60d"


def test_rolling_autocorrelation_nan_at_start():
    ret = _make_returns()
    result = rolling_autocorrelation(ret, lag=1, window=60)
    assert result.iloc[:59].isna().all()


def test_rolling_autocorrelation_bounded():
    """Autocorrelation must be in [-1, 1]."""
    ret = _make_returns()
    result = rolling_autocorrelation(ret, lag=1, window=60)
    valid = result.dropna()
    assert (valid >= -1.01).all()
    assert (valid <= 1.01).all()


def test_rolling_autocorrelation_strongly_persistent_series():
    """AR(1) series with rho=0.9 should show high positive autocorrelation."""
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    # Generate AR(1) process with rho=0.9
    x = np.zeros(n)
    eps = rng.normal(0, 0.01, n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + eps[i]
    ret = pd.Series(x, index=idx)
    result = rolling_autocorrelation(ret, lag=1, window=60)
    # AR(1) with rho=0.9 should produce consistently positive rolling autocorr
    assert result.dropna().mean() > 0.5


def test_rolling_autocorrelation_different_lags():
    ret = _make_returns()
    r1 = rolling_autocorrelation(ret, lag=1, window=60)
    r5 = rolling_autocorrelation(ret, lag=5, window=60)
    assert r1.name != r5.name
    assert not r1.equals(r5)
