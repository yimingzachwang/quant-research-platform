"""Trend features: moving averages, crossover signals, trend strength."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    result = close.rolling(window).mean()
    result.name = f"sma_{window}"
    return result


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    result = close.ewm(span=span, adjust=False).mean()
    result.name = f"ema_{span}"
    return result


def sma_crossover(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """SMA crossover signal.

    Returns +1 when fast SMA > slow SMA, -1 when below, 0 when equal.
    """
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()
    result = pd.Series(
        np.sign(fast_sma - slow_sma),
        index=close.index,
        name=f"sma_cross_{fast}_{slow}",
    )
    return result


def trend_persistence(
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling fraction of positive-return days within the window.

    Measures directional consistency: the proportion of trading days in the
    look-back window on which the asset closed higher than the previous day.
    A value of 0.5 indicates no directional persistence (random); above 0.5
    indicates upside consistency; below 0.5 indicates downside persistence.

    This is distinct from trend_strength (R² directional correlation) and from
    momentum (raw return over the window): it captures *how consistently* each
    day contributed to the trend, not its magnitude.

    Args:
        close: Price series.
        window: Look-back window in trading days.
    """
    daily_ret = close.pct_change()
    result = (daily_ret > 0).rolling(window).mean()
    result.name = f"trend_persist_{window}d"
    return result


def breakout_strength(
    close: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Distance of current price from rolling N-period high, normalised by peak.

    Returns (close / rolling_max) - 1, which is in (−∞, 0].
    A value near 0 indicates the price is at or near its rolling N-period high
    (breakout / momentum continuation regime).  A large negative value indicates
    the asset is well below its recent range top.

    Compared to drawdown_distance (which uses a longer window to measure extended
    stress), breakout_strength uses a short window to capture near-term price
    momentum relative to its recent range.

    Args:
        close: Price series.
        window: Look-back window for the rolling maximum.
    """
    rolling_max = close.rolling(window).max()
    denom = rolling_max.replace(0, float("nan"))
    result = (close / denom) - 1.0
    result.name = f"breakout_{window}d"
    return result


def trend_strength(close: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between price and time index over ``window`` periods.

    Ranges from -1 (perfect downtrend) to +1 (perfect uptrend).
    A value near 0 indicates a sideways or noisy regime.
    """
    t = pd.Series(np.arange(len(close)), index=close.index, dtype=float)

    def _corr(x: pd.Series) -> float:
        t_win = t.loc[x.index]
        if x.std() == 0 or t_win.std() == 0:
            return 0.0
        return float(x.corr(t_win))

    result = close.rolling(window).apply(_corr, raw=False)
    result.name = f"trend_strength_{window}d"
    return result
