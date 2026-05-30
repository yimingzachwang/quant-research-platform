"""Pure functions for portfolio turnover analysis of ML-driven strategies.

All functions are stateless — no fitting, no I/O, no side effects.

Note: src.visualization.diagnostics.compute_turnover computes the same
per-period turnover series for portfolio backtest output.  The functions
here are in the ML diagnostics namespace and operate on arbitrary weight
DataFrames (e.g., weights produced by MLStrategy or signal functions)
without requiring a full backtest run.
"""

from __future__ import annotations

import pandas as pd


def signal_turnover(weights: pd.DataFrame) -> pd.Series:
    """Per-period portfolio turnover as sum of absolute weight changes.

    Turnover_t = Σ_i |w_{i,t} - w_{i,t-1}|

    The first row is NaN (no prior weights to diff against).
    NaN weights are treated as 0 (flat) before differencing.

    Args:
        weights: Date × Asset weight DataFrame.

    Returns:
        pd.Series of per-period turnover values (first row is NaN).
    """
    w = weights.fillna(0.0)
    result = w.diff().abs().sum(axis=1)
    if len(result) > 0:
        result.iloc[0] = float("nan")  # no prior period for the first row
    result.name = "turnover"
    return result


def average_turnover(weights: pd.DataFrame) -> float:
    """Mean per-period turnover over the full history.

    Excludes the first row (NaN) produced by signal_turnover.

    Args:
        weights: Date × Asset weight DataFrame.

    Returns:
        Mean turnover as a float.  Returns float("nan") if fewer than
        2 rows are present (no valid diffs).
    """
    to = signal_turnover(weights).dropna()
    if len(to) == 0:
        return float("nan")
    return float(to.mean())


def turnover_by_split(
    split_weights: list[pd.DataFrame],
) -> pd.DataFrame:
    """Per-split turnover summary across walk-forward test windows.

    Args:
        split_weights: List of Date × Asset weight DataFrames, one per
                       walk-forward test window.  The index of each
                       DataFrame should cover only the test period.

    Returns:
        DataFrame with one row per split (0-indexed) and columns:
        mean_turnover, max_turnover, std_turnover, n_periods.
        Empty DataFrame if split_weights is empty.
    """
    if not split_weights:
        return pd.DataFrame(
            columns=["mean_turnover", "max_turnover", "std_turnover", "n_periods"]
        )

    rows = []
    for i, w in enumerate(split_weights):
        to = signal_turnover(w).dropna()
        if len(to) == 0:
            rows.append({
                "split": i,
                "mean_turnover": float("nan"),
                "max_turnover": float("nan"),
                "std_turnover": float("nan"),
                "n_periods": 0,
            })
        else:
            rows.append({
                "split": i,
                "mean_turnover": float(to.mean()),
                "max_turnover": float(to.max()),
                "std_turnover": float(to.std(ddof=1)) if len(to) > 1 else float("nan"),
                "n_periods": int(len(to)),
            })

    return pd.DataFrame(rows).set_index("split")
