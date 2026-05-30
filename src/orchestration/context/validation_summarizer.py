"""Summarizes walk-forward validation data for LLM context."""

from __future__ import annotations

from typing import Any

from src.orchestration.context.context_schema import (
    OOS_HIT_RATE_STRONG,
    OOS_HIT_RATE_WEAK,
)


def summarize_validation(split_metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Convert split_metrics.json into a structured validation summary."""
    if not split_metrics:
        return {"available": False}

    summary = split_metrics.get("summary", {})
    splits = split_metrics.get("splits", [])

    n_splits = summary.get("n_splits", len(splits))
    mean_sharpe = summary.get("mean_sharpe")
    std_sharpe = summary.get("std_sharpe")
    hit_rate = summary.get("hit_rate_positive_sharpe")
    mean_ret = summary.get("mean_annualized_return")
    worst_dd = summary.get("worst_max_drawdown")

    # split records use "sharpe_ratio" (the OOS test-window Sharpe)
    sharpe_vals = [s.get("sharpe_ratio") for s in splits if s.get("sharpe_ratio") is not None]
    negative_splits = sum(1 for v in sharpe_vals if v < 0)

    return {
        "available": True,
        "n_splits": n_splits,
        "mean_oos_sharpe": _round(mean_sharpe),
        "std_oos_sharpe": _round(std_sharpe),
        "hit_rate_positive_sharpe_pct": _pct(hit_rate),
        "mean_oos_return_pct": _pct(mean_ret),
        "worst_split_drawdown_pct": _pct(worst_dd),
        "n_negative_sharpe_splits": negative_splits,
        "consistency_tier": _consistency_tier(hit_rate),
        "per_split_sharpes": [_round(v) for v in sharpe_vals],
    }


def _pct(v: float | None) -> str | None:
    if v is None:
        return None
    return f"{v * 100:.2f}%"


def _round(v: float | None) -> float | None:
    if v is None:
        return None
    return round(v, 3)


def _consistency_tier(hit_rate: float | None) -> str:
    if hit_rate is None:
        return "unknown"
    if hit_rate >= OOS_HIT_RATE_STRONG:
        return "strong"
    if hit_rate >= OOS_HIT_RATE_WEAK:
        return "moderate"
    return "weak"
