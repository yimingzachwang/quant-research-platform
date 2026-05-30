"""Volatility features for return series."""

from __future__ import annotations

import math

import pandas as pd

_TRADING_DAYS_PER_YEAR = 252


def rolling_volatility(
    returns: pd.Series,
    window: int,
    annualize: bool = True,
) -> pd.Series:
    """Rolling standard deviation of returns.

    Args:
        returns: Period return series (e.g. from compute_returns).
        window: Look-back window in periods.
        annualize: Multiply by sqrt(252) when True.
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * math.sqrt(_TRADING_DAYS_PER_YEAR)
    vol.name = f"vol_{window}d{'_ann' if annualize else ''}"
    return vol


def ewm_volatility(
    returns: pd.Series,
    span: int,
    annualize: bool = True,
) -> pd.Series:
    """Exponentially weighted moving volatility.

    Args:
        returns: Period return series.
        span: EWM span parameter (half-life ≈ span / 2).
        annualize: Multiply by sqrt(252) when True.
    """
    vol = returns.ewm(span=span, adjust=False).std()
    if annualize:
        vol = vol * math.sqrt(_TRADING_DAYS_PER_YEAR)
    vol.name = f"ewm_vol_{span}span{'_ann' if annualize else ''}"
    return vol


def downside_volatility(
    returns: pd.Series,
    window: int,
    annualize: bool = True,
    threshold: float = 0.0,
) -> pd.Series:
    """Rolling downside volatility (semi-deviation below threshold).

    Measures the volatility of returns that fall below *threshold*.
    Used as a risk-regime indicator distinguishing downside risk from
    symmetric volatility.

    Args:
        returns: Period return series.
        window: Look-back window in periods.
        annualize: Multiply by sqrt(252) when True.
        threshold: Returns below this value count as downside (default 0).
    """
    downside = returns.clip(upper=threshold) - threshold

    def _semi_std(x: "pd.Series") -> float:
        neg = x[x < 0]
        return float(neg.std()) if len(neg) >= 2 else float("nan")

    vol = downside.rolling(window).apply(_semi_std, raw=False)
    if annualize:
        vol = vol * math.sqrt(_TRADING_DAYS_PER_YEAR)
    vol.name = f"downside_vol_{window}d"
    return vol


def vol_of_vol(
    returns: pd.Series,
    vol_window: int = 21,
    meta_window: int = 63,
    annualize: bool = True,
) -> pd.Series:
    """Volatility-of-volatility: rolling std of rolling volatility.

    Captures regime transitions — high vol-of-vol indicates an unstable or
    transitioning volatility regime.

    Args:
        returns: Period return series.
        vol_window: Window for inner rolling volatility estimate.
        meta_window: Window for the outer rolling std of volatility.
        annualize: Annualise the inner vol before computing its std.
    """
    inner = rolling_volatility(returns, vol_window, annualize=annualize)
    result = inner.rolling(meta_window).std()
    result.name = f"vol_of_vol_{vol_window}_{meta_window}"
    return result


def vol_compression(
    returns: pd.Series,
    short_window: int = 21,
    long_window: int = 63,
    annualize: bool = True,
) -> pd.Series:
    """Ratio of short-term realised vol to long-term realised vol.

    Values below 1.0 indicate a compressed-volatility regime — realised vol has
    contracted relative to its recent baseline.  Values above 1.0 indicate vol
    expansion.  Compression followed by rapid expansion is a classic breakout
    regime precursor.

    Args:
        returns: Period return series.
        short_window: Window for the short-term vol estimate (numerator).
        long_window: Window for the long-term vol estimate (denominator).
        annualize: Annualise both vol estimates before taking the ratio.
    """
    vol_short = rolling_volatility(returns, short_window, annualize=annualize)
    vol_long = rolling_volatility(returns, long_window, annualize=annualize)
    denom = vol_long.replace(0, float("nan"))
    result = vol_short / denom
    result.name = f"vol_compress_{short_window}_{long_window}"
    return result


def drawdown_distance(
    close: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Distance of current price from rolling N-period peak, normalised by peak.

    Returns (close - rolling_max) / rolling_max, which is in (−∞, 0].
    A value of 0 means the price is at its rolling N-period high.
    A value of −0.20 means the price is 20% below its rolling N-period high.

    This is a stress-state positioning indicator: assets with large negative
    values are in extended drawdowns relative to their recent price history.

    Args:
        close: Price series.
        window: Look-back window for the rolling maximum.
    """
    rolling_max = close.rolling(window).max()
    denom = rolling_max.replace(0, float("nan"))
    result = (close - denom) / denom
    result.name = f"drawdown_dist_{window}d"
    return result


def vol_percentile(
    returns: pd.Series,
    vol_window: int = 21,
    lookback: int = 252,
) -> pd.Series:
    """Rolling percentile rank of realised volatility in its own history.

    Returns a value in [0, 1].  High values (> 0.8) indicate elevated
    volatility relative to recent history; low values (< 0.2) indicate
    compressed volatility.

    Args:
        returns: Period return series.
        vol_window: Window for inner rolling volatility estimate.
        lookback: Look-back window for the percentile ranking.
    """
    inner = rolling_volatility(returns, vol_window, annualize=True)

    def _pctrank(x: "pd.Series") -> float:
        current = x.iloc[-1]
        if pd.isna(current) or len(x) <= 1:
            return float("nan")
        prior = x.iloc[:-1].dropna()
        if len(prior) == 0:
            return 0.5
        return float((prior < current).sum() / len(prior))

    result = inner.rolling(lookback).apply(_pctrank, raw=False)
    result.name = f"vol_pct_{vol_window}_{lookback}"
    return result
