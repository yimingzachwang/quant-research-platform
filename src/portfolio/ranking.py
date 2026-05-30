"""Cross-sectional ranking and selection utilities.

Operates on Date × Asset DataFrames.  All ranking is row-wise (per date),
so each row represents a cross-sectional snapshot of the universe.
"""

from __future__ import annotations

import pandas as pd


def rank_assets(
    df: pd.DataFrame,
    ascending: bool = False,
) -> pd.DataFrame:
    """Cross-sectional percentile rank per date.

    Args:
        df: Date × Asset feature DataFrame (e.g. momentum scores).
        ascending: If True, lower values get higher ranks.  Default False
            (higher feature value → higher rank), which is correct for
            momentum: high momentum → high rank.

    Returns:
        Date × Asset DataFrame with ranks in [0, 1].
        NaN inputs propagate as NaN ranks (asset is unrankable that period).
    """
    # API contract: ascending=False (default) → higher value = higher percentile rank (→ 1.0).
    # pandas rank(ascending=True) assigns rank n to the largest value so pct=n/n=1.0.
    # We therefore invert the user-facing flag before passing to pandas.
    return df.rank(axis=1, pct=True, ascending=not ascending, na_option="keep")


def select_top_n(
    ranks: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Boolean selection mask for the top-N assets by rank each day.

    Ties are broken by selecting the asset with the higher ordinal rank
    (method='first', consistent with pandas default).  If fewer than N
    assets have valid ranks on a given date, all valid assets are selected.

    Args:
        ranks: Date × Asset rank DataFrame from rank_assets().
        n: Number of assets to select per period.

    Returns:
        Date × Asset boolean DataFrame.  True = selected.
    """
    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)

    # Descending rank: rank 1 = highest value
    ordinal = ranks.rank(axis=1, ascending=False, method="first", na_option="keep")
    mask = ordinal <= n
    # Propagate NaN positions as False (unrankable = not selected)
    return mask.fillna(False).astype(bool)


def select_bottom_n(
    ranks: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Boolean selection mask for the bottom-N assets by rank each day.

    Useful for short-side selection in long/short strategies.

    Args:
        ranks: Date × Asset rank DataFrame from rank_assets().
        n: Number of assets to select per period.
    """
    if n < 1:
        msg = f"n must be >= 1, got {n}"
        raise ValueError(msg)

    ordinal = ranks.rank(axis=1, ascending=True, method="first", na_option="keep")
    mask = ordinal <= n
    return mask.fillna(False).astype(bool)
