"""Deterministic text summary of performance metrics.

Produces a structured dict (not prose) that the LLM can reason about
without needing to interpret raw numbers.
"""

from __future__ import annotations

from typing import Any

from src.orchestration.context.context_schema import (
    DRAWDOWN_ELEVATED,
    DRAWDOWN_SEVERE,
    SHARPE_EXCELLENT,
    SHARPE_GOOD,
    SHARPE_WEAK,
)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a structured performance summary dict for LLM context."""
    if not metrics:
        return {}

    sharpe = metrics.get("sharpe_ratio")
    ret = metrics.get("annualized_return")
    vol = metrics.get("annualized_volatility")
    dd = metrics.get("max_drawdown")
    calmar = metrics.get("calmar_ratio")
    hit = metrics.get("hit_rate")

    summary: dict[str, Any] = {
        "sharpe_ratio": sharpe,
        "annualized_return_pct": _pct(ret),
        "annualized_volatility_pct": _pct(vol),
        "max_drawdown_pct": _pct(dd),
        "calmar_ratio": _round(calmar),
        "hit_rate_pct": _pct(hit),
        "sharpe_tier": _sharpe_tier(sharpe),
        "drawdown_severity": _drawdown_severity(dd),
        "return_to_vol": _round(ret / vol) if ret is not None and vol else None,
    }
    return {k: v for k, v in summary.items() if v is not None}


def _pct(v: float | None) -> str | None:
    if v is None:
        return None
    return f"{v * 100:.2f}%"


def _round(v: float | None, ndigits: int = 3) -> float | None:
    if v is None:
        return None
    return round(v, ndigits)


def _sharpe_tier(sharpe: float | None) -> str:
    if sharpe is None:
        return "unknown"
    if sharpe >= SHARPE_EXCELLENT:
        return "excellent"
    if sharpe >= SHARPE_GOOD:
        return "good"
    if sharpe > SHARPE_WEAK:
        return "weak"
    return "negative"


def _drawdown_severity(dd: float | None) -> str:
    if dd is None:
        return "unknown"
    if dd <= DRAWDOWN_SEVERE:
        return "severe"
    if dd <= DRAWDOWN_ELEVATED:
        return "elevated"
    return "moderate"
