"""Panel (Date × Asset) feature computation utilities.

Each function takes a Date × Asset DataFrame and returns a Date × Asset
DataFrame of the same shape.  Column names (symbols) are preserved.

These are vectorized wrappers around the single-asset feature functions,
applied column-wise so the logic stays DRY and consistent.
"""

from __future__ import annotations

import math

import pandas as pd


def universe_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple period returns for every asset.  First row is NaN."""
    return prices.pct_change()


def universe_momentum(
    prices: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Cross-sectional momentum: close_t / close_{t-window} - 1.

    Args:
        prices: Date × Asset close price DataFrame.
        window: Look-back window in trading days.
    """
    return prices / prices.shift(window) - 1.0


def universe_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 63,
    annualize: bool = True,
) -> pd.DataFrame:
    """Rolling realized volatility for every asset.

    Args:
        returns: Date × Asset return DataFrame.
        window: Rolling window in trading days.
        annualize: Multiply by sqrt(252) when True.
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * math.sqrt(252)
    return vol


def universe_rolling_zscore(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Rolling z-score applied column-wise.  NaN during warm-up."""
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    return (df - mean) / std.replace(0.0, float("nan"))
