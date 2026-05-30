"""Rolling window utilities: z-score, rank, min-max normalization, and higher-moment features."""

from __future__ import annotations

import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    result = (series - mean) / std
    result.name = f"{series.name}_zscore_{window}d" if series.name else f"zscore_{window}d"
    return result


def rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank within window, result in [0, 1].

    Uses the fraction of window observations strictly less than the current value
    (ascending rank normalised to [0, 1]).
    """

    def _rank(x: pd.Series) -> float:
        current = x.iloc[-1]
        return float((x < current).sum() / (len(x) - 1)) if len(x) > 1 else 0.5

    result = series.rolling(window).apply(_rank, raw=False)
    result.name = f"{series.name}_rank_{window}d" if series.name else f"rank_{window}d"
    return result


def rolling_minmax(series: pd.Series, window: int) -> pd.Series:
    """Rolling min-max normalization: (x - rolling_min) / (rolling_max - rolling_min).

    Returns NaN when rolling_max == rolling_min (flat window).
    """
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()
    denom = rolling_max - rolling_min
    result = (series - rolling_min) / denom.replace(0, float("nan"))
    result.name = (
        f"{series.name}_minmax_{window}d" if series.name else f"minmax_{window}d"
    )
    return result


def bollinger_distance(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Signed distance of price from Bollinger Band midpoint, normalised by band width.

    Returns (close - SMA) / (n_std * rolling_std), i.e. the number of band
    half-widths the price is above/below the rolling mean.  A value of +1
    means price is exactly at the upper band; -1 at the lower band.

    Args:
        close: Price series.
        window: Look-back window for the SMA and rolling std.
        n_std: Number of standard deviations for the band width.
    """
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    denom = (n_std * std).replace(0, float("nan"))
    result = (close - sma) / denom
    result.name = f"bollinger_{window}d"
    return result


def rolling_skewness(series: pd.Series, window: int) -> pd.Series:
    """Rolling skewness of a series over ``window`` periods.

    Positive values indicate a right-skewed distribution (fat upper tail);
    negative values indicate a left-skewed distribution.  Useful for
    detecting distributional regime shifts and fat-tail exposure.
    """
    result = series.rolling(window).skew()
    result.name = f"skew_{window}d"
    return result


def rolling_autocorrelation(series: pd.Series, lag: int = 1, window: int = 60) -> pd.Series:
    """Rolling lag-1 (or lag-k) autocorrelation over ``window`` periods.

    High positive autocorrelation indicates trend/momentum persistence;
    negative autocorrelation indicates mean-reverting behaviour.

    Args:
        series: Input series (typically returns).
        lag: Autocorrelation lag (default 1).
        window: Rolling window length.
    """
    def _autocorr(x: pd.Series) -> float:
        if len(x) < lag + 2:
            return float("nan")
        return float(pd.Series(x).autocorr(lag=lag))

    result = series.rolling(window).apply(_autocorr, raw=False)
    result.name = f"autocorr_{lag}_{window}d"
    return result
