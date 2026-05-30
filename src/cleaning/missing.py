"""Missing value utilities for research-safe gap filling."""

from __future__ import annotations

import pandas as pd


def forward_fill_limited(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Forward-fill NaN values up to ``limit`` consecutive periods.

    Gaps longer than ``limit`` periods remain NaN, making the fill visible
    and auditable rather than silently propagating stale values.

    Args:
        df: Input DataFrame (any column dtypes).
        limit: Maximum number of consecutive NaNs to fill.
    """
    if limit < 1:
        msg = f"limit must be >= 1, got {limit}"
        raise ValueError(msg)
    return df.ffill(limit=limit)
