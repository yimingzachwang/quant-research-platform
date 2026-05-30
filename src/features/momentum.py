"""Momentum features for price series."""

from __future__ import annotations

import pandas as pd


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Price momentum over ``window`` periods: close_t / close_{t-window} - 1.

    Named output: ``momentum_{window}d``.
    """
    result = close / close.shift(window) - 1
    result.name = f"momentum_{window}d"
    return result


def momentum_20d(close: pd.Series) -> pd.Series:
    """1-month momentum (20 trading days)."""
    return momentum(close, 20)


def momentum_60d(close: pd.Series) -> pd.Series:
    """3-month momentum (60 trading days)."""
    return momentum(close, 60)


def momentum_252d(close: pd.Series) -> pd.Series:
    """12-month momentum (252 trading days)."""
    return momentum(close, 252)


def risk_adjusted_momentum(
    close: pd.Series,
    mom_window: int = 252,
    vol_window: int = 63,
) -> pd.Series:
    """Momentum normalised by realised volatility — a Sharpe-like signal.

    Divides trailing mom_window return by rolling vol_window annualised
    volatility, yielding a dimensionless score that rewards high-momentum
    assets that achieve their returns efficiently (low vol).  Assets with
    equal raw momentum but lower vol receive a higher score.

    This exposes a distinct information dimension from raw momentum: the model
    can learn whether the *quality* of momentum (consistency, vol-adjusted)
    predicts relative performance better than its raw magnitude.

    Args:
        close: Price series.
        mom_window: Look-back window for trailing return.
        vol_window: Look-back window for realised vol (annualised).
    """
    import math

    raw_mom = momentum(close, mom_window)
    returns = close.pct_change()
    ann_vol = returns.rolling(vol_window).std() * math.sqrt(252)
    denom = ann_vol.replace(0, float("nan"))
    result = raw_mom / denom
    result.name = f"sharpe_mom_{mom_window}d"
    return result
