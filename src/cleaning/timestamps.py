"""Timestamp index cleaning utilities."""

from __future__ import annotations

import pandas as pd


def sort_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` sorted ascending by its DatetimeIndex."""
    return df.sort_index()


def remove_duplicate_timestamps(
    df: pd.DataFrame,
    keep: str = "last",
) -> pd.DataFrame:
    """Return a copy of ``df`` with duplicate index entries removed.

    Args:
        df: DataFrame with a DatetimeIndex.
        keep: Which duplicate to retain — ``"first"`` or ``"last"``.
    """
    if keep not in {"first", "last"}:
        msg = f"keep must be 'first' or 'last', got {keep!r}"
        raise ValueError(msg)
    return df[~df.index.duplicated(keep=keep)]
