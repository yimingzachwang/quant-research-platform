"""Numeric cleaning utilities."""

from __future__ import annotations

import pandas as pd


def replace_inf(df: pd.DataFrame, value: float = float("nan")) -> pd.DataFrame:
    """Replace +inf and -inf with ``value`` (default NaN) in all numeric columns.

    Non-numeric columns are left unchanged.
    """
    return df.replace([float("inf"), float("-inf")], value)
