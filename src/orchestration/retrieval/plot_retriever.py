"""Retrieves plot paths and metadata from the plots/ directory."""

from __future__ import annotations

from pathlib import Path

from src.orchestration.api.schemas import PlotMetadata
from src.orchestration.utils.filesystem import plot_index_path, plots_dir
from src.orchestration.utils.serialization import load_json


def get_plot_index(
    experiment_name: str,
    base: Path | str | None = None,
) -> list[PlotMetadata]:
    """Load plot_index.json and return typed PlotMetadata list."""
    raw = load_json(plot_index_path(experiment_name, base))
    if not raw or not isinstance(raw, list):
        return []
    pdir = plots_dir(experiment_name, base)
    result: list[PlotMetadata] = []
    for entry in raw:
        name = entry.get("name", "")
        path = str(pdir / f"{name}.png") if name else None
        result.append(PlotMetadata(
            name=name,
            group=entry.get("group", ""),
            importance=entry.get("importance", ""),
            caption=entry.get("caption", ""),
            path=path,
        ))
    return result


def list_plot_stems(
    experiment_name: str,
    base: Path | str | None = None,
) -> list[str]:
    """Return stems of all PNG files in the plots/ directory."""
    pdir = plots_dir(experiment_name, base)
    if not pdir.exists():
        return []
    return sorted(p.stem for p in pdir.glob("*.png"))


def plot_exists(
    experiment_name: str,
    stem: str,
    base: Path | str | None = None,
) -> bool:
    pdir = plots_dir(experiment_name, base)
    return (pdir / f"{stem}.png").exists()


def get_primary_plots(
    experiment_name: str,
    base: Path | str | None = None,
) -> list[PlotMetadata]:
    """Return only primary-importance plots from the plot index."""
    return [p for p in get_plot_index(experiment_name, base) if p.importance == "primary"]
