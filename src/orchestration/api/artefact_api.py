"""High-level artefact access API.

Provides typed, named access to every experiment artefact group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.api.schemas import ArtefactMetadata, PlotMetadata
from src.orchestration.retrieval.artefact_retriever import list_artefacts, retrieve
from src.orchestration.retrieval.diagnostics_retriever import load_all_diagnostics
from src.orchestration.retrieval.plot_retriever import (
    get_plot_index,
    get_primary_plots,
    list_plot_stems,
)


def get_artefact(
    experiment_name: str,
    key: str,
    base: Path | str | None = None,
) -> Any:
    """Load a single artefact by key; returns None if absent."""
    return retrieve(experiment_name, key, base)


def get_all_diagnostics(
    experiment_name: str,
    base: Path | str | None = None,
) -> dict[str, Any]:
    """Load every diagnostics and research JSON into one dict."""
    return load_all_diagnostics(experiment_name, base)


def list_experiment_artefacts(
    experiment_name: str,
    base: Path | str | None = None,
) -> list[ArtefactMetadata]:
    """Return ArtefactMetadata for all known artefacts of this experiment."""
    return list_artefacts(experiment_name, base)


def get_plots(
    experiment_name: str,
    base: Path | str | None = None,
    primary_only: bool = False,
) -> list[PlotMetadata]:
    """Return plot metadata from plot_index.json."""
    if primary_only:
        return get_primary_plots(experiment_name, base)
    return get_plot_index(experiment_name, base)


def list_plot_names(
    experiment_name: str,
    base: Path | str | None = None,
) -> list[str]:
    """Return stems of all PNG files in the plots/ directory."""
    return list_plot_stems(experiment_name, base)
