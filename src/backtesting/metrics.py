"""Standard performance metrics for strategy evaluation.

All functions accept a pd.Series of period returns (daily by default).
Annualization assumes 252 trading days unless overridden.
NaN periods are dropped before computation.
"""

from __future__ import annotations

import math

import pandas as pd

_TRADING_DAYS = 252


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = _TRADING_DAYS,
) -> float:
    """Compound annual growth rate (CAGR).

    Formula: (1 + total_return) ^ (periods_per_year / n) - 1
    Returns NaN for empty or all-NaN input.
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return float("nan")
    total = (1.0 + r).prod()
    return float(total ** (periods_per_year / n) - 1.0)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = _TRADING_DAYS,
) -> float:
    """Annualized standard deviation of period returns.

    Returns NaN for fewer than 2 observations.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std() * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio.

    Excess returns are computed per-period: r_t - (risk_free_rate / periods_per_year).
    Returns NaN when volatility is zero or fewer than 2 observations.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_per_period = risk_free_rate / periods_per_year
    excess = r - rf_per_period
    vol = excess.std()
    if vol == 0.0:
        return float("nan")
    return float(excess.mean() / vol * math.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown expressed as a negative fraction.

    Example: -0.25 means a 25% drawdown.
    Returns 0.0 for empty or all-NaN input (no drawdown possible).
    """
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    equity = (1.0 + r).cumprod()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = _TRADING_DAYS,
) -> float:
    """Annualized return divided by absolute max drawdown.

    Returns NaN when max drawdown is zero (no losses — denominator undefined).
    """
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return float("nan")
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    return float(ann_ret / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-NaN periods with a strictly positive return.

    Returns NaN for empty or all-NaN input.
    """
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).sum() / len(r))


def compute_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_DAYS,
) -> dict[str, float]:
    """Compute all standard metrics and return as a flat dict.

    Keys match the individual function names for easy DataFrame construction.
    """
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
    }
