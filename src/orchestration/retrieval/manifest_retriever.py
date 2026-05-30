"""Reads report manifests from reports/markdown/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.utils.filesystem import report_manifest_path
from src.orchestration.utils.serialization import load_json


def load_manifest(
    experiment_name: str,
    reports_base: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load the report manifest JSON for an experiment."""
    path = report_manifest_path(experiment_name, reports_base)
    return load_json(path)


def get_rendered_sections(
    experiment_name: str,
    reports_base: Path | str | None = None,
) -> list[str]:
    """Return the list of rendered section names from the manifest."""
    manifest = load_manifest(experiment_name, reports_base)
    if not manifest:
        return []
    return manifest.get("sections_rendered", [])


def get_manifest_metrics(
    experiment_name: str,
    reports_base: Path | str | None = None,
) -> dict[str, float]:
    """Return the metrics_summary block from the manifest."""
    manifest = load_manifest(experiment_name, reports_base)
    if not manifest:
        return {}
    return manifest.get("metrics_summary", {})


def get_figure_hierarchy(
    experiment_name: str,
    reports_base: Path | str | None = None,
) -> dict[str, list[str]]:
    """Return the figure_hierarchy dict (primary / secondary) from manifest."""
    manifest = load_manifest(experiment_name, reports_base)
    if not manifest:
        return {}
    return manifest.get("figure_hierarchy", {})
