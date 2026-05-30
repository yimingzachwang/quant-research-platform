"""Thin wrapper around src.experiments.registry for the orchestration layer.

Adds listing, filtering, and summary-enrichment without touching the
underlying registry implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.api.schemas import ExperimentSummary
from src.orchestration.utils.filesystem import (
    experiments_root,
    list_experiments,
    metadata_path,
    metrics_path,
)
from src.orchestration.utils.serialization import load_json


def list_all(base: Path | str | None = None) -> list[str]:
    """Return names of all experiment directories that have a metadata.json."""
    return list_experiments(base)


def find_by_tag(tag: str, base: Path | str | None = None) -> list[str]:
    """Return experiment names whose tags include the given tag.

    Checks registry.json (canonical source for tags) first, then falls back
    to reading metadata.json for experiments that may not be in the registry.
    """
    # Load from registry.json if present
    registry_path = experiments_root(base) / "registry.json"
    registry_entries = load_json(registry_path)
    if isinstance(registry_entries, list):
        in_registry = {
            e["experiment_name"]
            for e in registry_entries
            if isinstance(e, dict) and tag in e.get("tags", [])
        }
        if in_registry:
            return sorted(in_registry)

    # Fallback: scan metadata.json for each experiment
    names = list_all(base)
    matches = []
    for name in names:
        meta = load_json(metadata_path(name, base))
        if meta and tag in meta.get("tags", []):
            matches.append(name)
    return matches


def find_by_strategy(pattern: str, base: Path | str | None = None) -> list[str]:
    """Return experiment names whose strategy_name contains pattern (case-insensitive)."""
    names = list_all(base)
    pat = pattern.lower()
    matches = []
    for name in names:
        meta = load_json(metadata_path(name, base))
        if meta and pat in meta.get("strategy_name", "").lower():
            matches.append(name)
    return matches


def get_summary(name: str, base: Path | str | None = None) -> ExperimentSummary | None:
    """Build an ExperimentSummary for one experiment from disk."""
    meta = load_json(metadata_path(name, base))
    if meta is None:
        return None
    metrics = load_json(metrics_path(name, base)) or {}

    from src.orchestration.utils.filesystem import experiment_root
    root = experiment_root(name, base)

    ml_diag_path = root / "diagnostics" / "ml_model_diagnostics.json"
    split_path = root / "diagnostics" / "split_metrics.json"

    return ExperimentSummary(
        experiment_name=name,
        strategy_name=meta.get("strategy_name", ""),
        created_at=meta.get("created_at", ""),
        tags=meta.get("tags", []),
        sharpe_ratio=metrics.get("sharpe_ratio"),
        annualized_return=metrics.get("annualized_return"),
        max_drawdown=metrics.get("max_drawdown"),
        has_ml=ml_diag_path.exists(),
        has_validation=split_path.exists(),
        artefact_root=str(root),
    )


def list_summaries(base: Path | str | None = None) -> list[ExperimentSummary]:
    """Return ExperimentSummary for every experiment on disk."""
    return [s for name in list_all(base) if (s := get_summary(name, base)) is not None]


def rank_by_sharpe(
    base: Path | str | None = None,
    descending: bool = True,
) -> list[ExperimentSummary]:
    """Return all summaries sorted by Sharpe ratio."""
    summaries = list_summaries(base)
    return sorted(
        summaries,
        key=lambda s: s.sharpe_ratio if s.sharpe_ratio is not None else float("-inf"),
        reverse=descending,
    )
