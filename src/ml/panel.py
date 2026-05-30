"""Cross-sectional panel utilities for multi-asset ML research.

Extends the single-asset ML pipeline to a pooled panel:
    build_panel_feature_matrix  — stack per-asset features into MultiIndex(date, asset)
    compute_cross_sectional_ic  — per-date Spearman IC across the asset cross-section

Dependency direction: this module imports from src.ml.feature_matrix only.
No I/O, no model fitting, no strategy logic.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd


def build_panel_feature_matrix(
    prices: pd.DataFrame,
    feature_fn_builder: Callable[[str], dict[str, Callable[[pd.DataFrame], pd.Series]]],
    tickers: list[str],
) -> pd.DataFrame:
    """Build a (Date × Asset, Features) panel feature matrix.

    For each ticker, calls feature_fn_builder(ticker) to get per-ticker feature
    callables, then applies build_feature_matrix() to produce a Date-indexed
    feature DataFrame.  All ticker DataFrames are stacked into a MultiIndex
    DataFrame with index levels (date, asset).

    NaN rows are NOT dropped here — call dropna() before model consumption.

    Args:
        prices: Date × Asset price DataFrame (DatetimeIndex columns are tickers).
        feature_fn_builder: Callable that takes a ticker name and returns a dict
            of {feature_name: fn(prices: pd.DataFrame) -> pd.Series}.
        tickers: Ordered list of asset tickers to include in the panel.

    Returns:
        DataFrame with MultiIndex(date, asset) index and one column per feature.
        Rows are sorted by (date, asset).  Returns an empty DataFrame if
        tickers is empty.
    """
    from src.ml.feature_matrix import build_feature_matrix

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        feature_fns = feature_fn_builder(ticker)
        feat_df = build_feature_matrix(prices, feature_fns)
        # Attach asset level — create MultiIndex(date, asset)
        feat_df = feat_df.copy()
        feat_df.index = pd.MultiIndex.from_arrays(
            [feat_df.index, [ticker] * len(feat_df)],
            names=["date", "asset"],
        )
        frames.append(feat_df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=0)
    panel.sort_index(inplace=True)
    return panel


def compute_cross_sectional_ic(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    min_assets: int = 3,
) -> pd.Series:
    """Compute per-date Spearman rank IC across the asset cross-section.

    For each date present in both pred_df and actual_df:
        IC(t) = Spearman(predicted_ranks_t, actual_ranks_t)

    Dates with fewer than min_assets non-null observations are skipped.

    Args:
        pred_df:   Date × Asset DataFrame of model prediction scores.
        actual_df: Date × Asset DataFrame of realised forward returns.
        min_assets: Minimum number of valid assets required per date.

    Returns:
        pd.Series of daily IC values indexed by date.  Empty if no dates
        meet the minimum asset threshold.
    """
    common_dates = pred_df.index.intersection(actual_df.index)
    ic_values: list[float] = []
    ic_dates: list[pd.Timestamp] = []

    for date in common_dates:
        p = pred_df.loc[date].dropna()
        a = actual_df.loc[date].dropna()
        common_assets = p.index.intersection(a.index)
        if len(common_assets) < min_assets:
            continue
        p_common = p.loc[common_assets]
        a_common = a.loc[common_assets]
        p_ranked = p_common.rank()
        a_ranked = a_common.rank()
        corr = float(p_ranked.corr(a_ranked))
        ic_values.append(corr)
        ic_dates.append(date)

    if not ic_dates:
        return pd.Series(dtype=float)

    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name="cross_sectional_ic")
