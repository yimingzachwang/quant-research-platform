"""Minimal portfolio-level utilities.

These operate on position or signal series produced by run_backtest().
No optimization, no constraint solving, no allocation framework.
"""

from __future__ import annotations

import pandas as pd


def compute_turnover(positions: pd.Series) -> pd.Series:
    """Absolute change in position each period.

    Equivalent to the ``turnover`` column produced by run_backtest().
    Useful when computing turnover independently of the engine.

    First period is NaN (no prior position to diff against).
    """
    result = positions.diff().abs()
    result.name = "turnover"
    return result


def compute_exposure(positions: pd.Series) -> pd.Series:
    """Absolute gross exposure each period: |position_t|."""
    result = positions.abs()
    result.name = "exposure"
    return result


def position_sizing(
    signal: pd.Series,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Scale a signal so the maximum absolute value equals ``max_leverage``.

    If the signal is all-zero or all-NaN, it is returned unchanged.

    Args:
        signal: Raw signal series.
        max_leverage: Target absolute cap (e.g. 1.0 = fully invested).
    """
    peak = signal.abs().max()
    if peak == 0.0 or pd.isna(peak):
        return signal.copy()
    result = signal / peak * max_leverage
    result.name = signal.name
    return result
