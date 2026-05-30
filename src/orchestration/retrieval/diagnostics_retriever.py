"""Specialized retrieval for diagnostics/ and research/ subdirectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.utils.filesystem import diagnostics_dir, research_dir
from src.orchestration.utils.serialization import load_json


def load_backtest_diagnostics(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(diagnostics_dir(name, base) / "backtest_diagnostics.json")


def load_ml_diagnostics(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(diagnostics_dir(name, base) / "ml_diagnostics.json")


def load_ml_model_diagnostics(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(diagnostics_dir(name, base) / "ml_model_diagnostics.json")


def load_split_metrics(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(diagnostics_dir(name, base) / "split_metrics.json")


def load_universe_coverage(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(diagnostics_dir(name, base) / "universe_coverage.json")


def load_wf_equity_curves(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(diagnostics_dir(name, base) / "wf_equity_curves.json")


def load_alignment_diagnostics(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(research_dir(name, base) / "alignment_diagnostics.json")


def load_feature_summary(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(research_dir(name, base) / "feature_summary.json")


def load_feature_families(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(research_dir(name, base) / "feature_families.json")


def load_feature_correlations(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(research_dir(name, base) / "feature_correlations.json")


def load_signal_transitions(
    name: str, base: Path | str | None = None
) -> dict[str, Any] | None:
    return load_json(research_dir(name, base) / "signal_transitions.json")


def load_all_diagnostics(
    name: str, base: Path | str | None = None
) -> dict[str, Any]:
    """Load every diagnostics and research JSON; omit missing files."""
    loaders = {
        "backtest_diagnostics": load_backtest_diagnostics,
        "ml_diagnostics": load_ml_diagnostics,
        "ml_model_diagnostics": load_ml_model_diagnostics,
        "split_metrics": load_split_metrics,
        "universe_coverage": load_universe_coverage,
        "wf_equity_curves": load_wf_equity_curves,
        "alignment_diagnostics": load_alignment_diagnostics,
        "feature_summary": load_feature_summary,
        "feature_families": load_feature_families,
        "feature_correlations": load_feature_correlations,
        "signal_transitions": load_signal_transitions,
    }
    result: dict[str, Any] = {}
    for key, loader in loaders.items():
        data = loader(name, base)
        if data is not None:
            result[key] = data
    return result
