"""Full-series normalization utilities (non-rolling, fit on entire input)."""

from __future__ import annotations

import pandas as pd


def zscore_normalize(series: pd.Series) -> pd.Series:
    """Z-score normalization using series mean and std.

    Not leakage-safe for time-series modelling — use rolling_zscore for that.
    Intended for diagnostics and cross-sectional normalization.
    """
    mean = series.mean()
    std = series.std()
    result = (series - mean) / std
    result.name = f"{series.name}_zscore" if series.name else "zscore"
    return result


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalization to [0, 1] using series min and max.

    Returns NaN for all values when series min == max (constant series).
    """
    lo = series.min()
    hi = series.max()
    denom = hi - lo
    if denom == 0:
        return pd.Series(float("nan"), index=series.index, name=series.name)
    result = (series - lo) / denom
    result.name = f"{series.name}_minmax" if series.name else "minmax"
    return result


def robust_normalize(series: pd.Series) -> pd.Series:
    """Robust normalization using median and IQR.

    More resistant to outliers than z-score.  Returns NaN when IQR == 0.
    """
    median = series.median()
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    iqr = q75 - q25
    if iqr == 0:
        return pd.Series(float("nan"), index=series.index, name=series.name)
    result = (series - median) / iqr
    result.name = f"{series.name}_robust" if series.name else "robust"
    return result
