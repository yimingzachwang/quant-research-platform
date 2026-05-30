"""Portfolio weight construction from selection masks.

No optimizers, no covariance matrices, no constraint solvers.

Functions take a boolean selection mask and return a weight DataFrame
where rows sum to 1 (or 0 when no assets are selected).
"""

from __future__ import annotations

import pandas as pd


def equal_weight(mask: pd.DataFrame) -> pd.DataFrame:
    """Assign equal weight to each selected asset.

    Weight per selected asset = 1 / N_selected.
    Rows with no selected assets receive weight 0 for all assets.

    Args:
        mask: Date × Asset boolean DataFrame from select_top_n().

    Returns:
        Date × Asset weight DataFrame.  Rows sum to 1 (or 0).
    """
    m = mask.astype(float)
    n_selected = m.sum(axis=1)
    # Avoid division by zero: rows with 0 selected stay 0
    weights = m.div(n_selected.replace(0.0, float("nan")), axis=0).fillna(0.0)
    return weights


def volatility_scaled(
    mask: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 63,
    max_weight: float = 0.5,
) -> pd.DataFrame:
    """Inverse-volatility weighting within the selected basket.

    For each selected asset:  raw_weight ∝ 1 / realized_vol
    Weights are normalised to sum to 1 per row, then capped at max_weight
    and renormalised once.

    Assets not selected by mask receive weight 0.
    Rows where all selected assets have zero or NaN volatility fall back
    to equal-weight for that row.

    Args:
        mask: Date × Asset boolean DataFrame.
        returns: Date × Asset return DataFrame (same index).
        window: Rolling volatility look-back window.
        max_weight: Maximum weight per asset after normalisation.
    """
    vol = returns.rolling(window).std()
    # Replace 0 vol with NaN so division is safe
    vol = vol.replace(0.0, float("nan"))

    inv_vol = 1.0 / vol
    # Zero out non-selected assets
    inv_vol = inv_vol.where(mask, other=0.0)

    row_sum = inv_vol.sum(axis=1).replace(0.0, float("nan"))
    weights = inv_vol.div(row_sum, axis=0).fillna(0.0)

    # Cap at max_weight and renormalize once
    weights = weights.clip(upper=max_weight)
    cap_sum = weights.sum(axis=1).replace(0.0, float("nan"))
    weights = weights.div(cap_sum, axis=0).fillna(0.0)

    return weights


def resample_weights_to_daily(
    weights: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill periodic weights (e.g. monthly) to a daily index.

    The weight decided at the end of period P is carried forward until the
    next rebalance date.  Combined with the 1-day lag in the backtest engine,
    this correctly represents: signal at end of month M → position from
    start of month M+1.

    Args:
        weights: Periodic weight DataFrame (e.g. month-end dates).
        daily_index: Target daily DatetimeIndex to reindex onto.

    Returns:
        Daily weight DataFrame forward-filled from the periodic weights.
        Dates before the first weight date have weight 0.
    """
    # Reindex: insert periodic weight rows into daily index, then ffill
    combined_index = daily_index.union(weights.index).sort_values()
    daily = weights.reindex(combined_index).ffill()
    daily = daily.reindex(daily_index).fillna(0.0)
    return daily
