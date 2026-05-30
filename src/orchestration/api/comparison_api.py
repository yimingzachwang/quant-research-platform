"""Experiment comparison API.

Wraps src.experiments.comparison where that module provides the comparison
logic; adds lightweight metric-table construction for LLM consumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.registry.experiment_registry import get_summary, list_summaries
from src.orchestration.api.schemas import ExperimentSummary


def compare_experiments(
    experiment_names: list[str],
    base: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return a list of metric rows for each named experiment.

    Suitable for tabular display or inclusion in an LLM context prompt.
    Experiments that cannot be loaded are silently omitted.
    """
    rows = []
    for name in experiment_names:
        summary = get_summary(name, base)
        if summary is None:
            continue
        rows.append(_summary_to_row(summary))
    return rows


def rank_experiments(
    base: Path | str | None = None,
    metric: str = "sharpe_ratio",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Return all experiment summaries ranked by a named metric."""
    summaries = list_summaries(base)

    def _key(s: ExperimentSummary) -> float:
        val = getattr(s, metric, None)
        if val is None:
            return float("-inf") if descending else float("inf")
        return val

    ranked = sorted(summaries, key=_key, reverse=descending)
    return [_summary_to_row(s) for s in ranked]


def diff_experiments(
    name_a: str,
    name_b: str,
    base: Path | str | None = None,
) -> dict[str, Any]:
    """Return a structured diff of key metrics between two experiments."""
    a = get_summary(name_a, base)
    b = get_summary(name_b, base)

    if a is None or b is None:
        missing = [n for n, s in [(name_a, a), (name_b, b)] if s is None]
        return {"error": f"Could not load experiments: {missing}"}

    metrics = ["sharpe_ratio", "annualized_return", "max_drawdown"]
    diffs: dict[str, Any] = {}
    for m in metrics:
        va = getattr(a, m, None)
        vb = getattr(b, m, None)
        diffs[m] = {
            name_a: va,
            name_b: vb,
            "delta": (vb - va) if (va is not None and vb is not None) else None,
        }
    return {
        "experiment_a": name_a,
        "experiment_b": name_b,
        "metric_diffs": diffs,
    }


def _summary_to_row(s: ExperimentSummary) -> dict[str, Any]:
    return {
        "experiment_name": s.experiment_name,
        "strategy_name": s.strategy_name,
        "created_at": s.created_at,
        "tags": s.tags,
        "sharpe_ratio": s.sharpe_ratio,
        "annualized_return": s.annualized_return,
        "max_drawdown": s.max_drawdown,
        "has_ml": s.has_ml,
        "has_validation": s.has_validation,
    }
