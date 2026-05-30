"""Pure diagnostic helper functions for portfolio analysis.

These functions compute statistics from weights/returns DataFrames and return
plain pandas objects. No plotting, no mutation of inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio turnover as sum of absolute weight changes.

    Args:
        weights: Date × Asset weight DataFrame.

    Returns:
        Series of per-day turnover values (NaN on first row).
    """
    return weights.diff().abs().sum(axis=1, skipna=False)


def compute_concentration_metrics(weights: pd.DataFrame) -> pd.DataFrame:
    """Per-row concentration statistics for a portfolio.

    Args:
        weights: Date × Asset weight DataFrame (values in [0, 1]).

    Returns:
        DataFrame with columns: hhi, max_weight, effective_n.
        - hhi: Herfindahl-Hirschman Index = Σ w_i²
        - max_weight: largest single-asset weight
        - effective_n: 1/HHI — effective number of assets held
    """
    w = weights.fillna(0.0)
    hhi = (w ** 2).sum(axis=1)
    max_weight = w.max(axis=1)
    effective_n = (1.0 / hhi).where(hhi > 0, other=np.nan)
    return pd.DataFrame(
        {"hhi": hhi, "max_weight": max_weight, "effective_n": effective_n},
        index=weights.index,
    )


def rolling_average_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling mean of all pairwise correlations across assets.

    For a portfolio with N assets this computes all N*(N-1)/2 pair rolling
    correlations and returns their mean at each time step.

    Args:
        returns: Date × Asset return DataFrame.
        window: Rolling window length in periods.

    Returns:
        Series of mean pairwise correlation, same index as returns.
        All-NaN if fewer than 2 columns are present.
    """
    cols = list(returns.columns)
    pairs = [(c1, c2) for i, c1 in enumerate(cols) for c2 in cols[i + 1 :]]
    if not pairs:
        return pd.Series(np.nan, index=returns.index, name="avg_pairwise_corr")
    rolling_corrs = pd.concat(
        [returns[c1].rolling(window).corr(returns[c2]) for c1, c2 in pairs],
        axis=1,
    )
    result = rolling_corrs.mean(axis=1)
    result.name = "avg_pairwise_corr"
    return result
