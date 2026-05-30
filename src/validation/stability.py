"""Stability analytics for walk-forward validation results.

These functions distill a WalkForwardResult into summary tables and statistics
useful for diagnosing overfitting, regime dependence, and parameter sensitivity.

All functions are pure: no mutation, no side effects, no plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.walk_forward import WalkForwardResult


# ---------------------------------------------------------------------------
# Per-split summary
# ---------------------------------------------------------------------------


def split_metrics_table(wf_result: WalkForwardResult) -> pd.DataFrame:
    """One row per split with metrics and date range columns.

    Returns:
        DataFrame indexed by split_index with columns: train_start,
        train_end, test_start, test_end, and all metric keys.
    """
    rows = []
    for sr in wf_result.splits:
        row: dict = {
            "split": sr.split.split_index,
            "train_start": sr.split.train_start,
            "train_end": sr.split.train_end,
            "test_start": sr.split.test_start,
            "test_end": sr.split.test_end,
        }
        row.update(sr.metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("split")


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def summarize_stability(wf_result: WalkForwardResult) -> dict[str, float]:
    """Aggregate statistics across all test splits.

    Returns:
        Dict with keys:
            n_splits, mean_sharpe, std_sharpe, hit_rate_positive_sharpe,
            mean_annualized_return, std_annualized_return,
            mean_max_drawdown, worst_max_drawdown.
    """
    table = split_metrics_table(wf_result)
    if table.empty:
        return {"n_splits": 0}

    sharpes = table["sharpe_ratio"].dropna()
    returns = table["annualized_return"].dropna()
    drawdowns = table["max_drawdown"].dropna()

    return {
        "n_splits": len(wf_result.splits),
        "mean_sharpe": float(sharpes.mean()),
        "std_sharpe": float(sharpes.std(ddof=1)) if len(sharpes) > 1 else float("nan"),
        "hit_rate_positive_sharpe": float((sharpes > 0).mean()),
        "mean_annualized_return": float(returns.mean()),
        "std_annualized_return": float(returns.std(ddof=1)) if len(returns) > 1 else float("nan"),
        "mean_max_drawdown": float(drawdowns.mean()),
        "worst_max_drawdown": float(drawdowns.min()),
    }


# ---------------------------------------------------------------------------
# Time series of Sharpe per split
# ---------------------------------------------------------------------------


def rolling_sharpe_by_split(wf_result: WalkForwardResult) -> pd.Series:
    """Sharpe ratio per split indexed by test_start date.

    Useful for time-series plots of out-of-sample Sharpe stability.
    """
    data = {
        sr.split.test_start: sr.metrics.get("sharpe_ratio", float("nan"))
        for sr in wf_result.splits
    }
    s = pd.Series(data, name="sharpe_ratio")
    s.index.name = "test_start"
    return s


# ---------------------------------------------------------------------------
# Multi-strategy parameter robustness
# ---------------------------------------------------------------------------


def parameter_robustness_summary(
    results: dict[str, WalkForwardResult],
    metric: str = "sharpe_ratio",
) -> pd.DataFrame:
    """Compare out-of-sample stability across multiple strategy configurations.

    Intended for sensitivity analysis — e.g. varying lookback or top_n.

    Args:
        results: Mapping of strategy label → WalkForwardResult.
        metric: Column in the per-split metrics table to analyse.

    Returns:
        DataFrame with one row per strategy and columns:
        mean, std, min, max, hit_rate_positive, n_splits.
    """
    rows: dict[str, dict] = {}
    for name, wf in results.items():
        table = split_metrics_table(wf)
        if table.empty or metric not in table.columns:
            rows[name] = {"mean": float("nan"), "std": float("nan"),
                          "min": float("nan"), "max": float("nan"),
                          "hit_rate_positive": float("nan"), "n_splits": 0}
            continue
        vals = table[metric].dropna()
        rows[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "hit_rate_positive": float((vals > 0).mean()),
            "n_splits": int(len(wf.splits)),
        }
    df = pd.DataFrame(rows).T
    df.index.name = "strategy"
    return df
