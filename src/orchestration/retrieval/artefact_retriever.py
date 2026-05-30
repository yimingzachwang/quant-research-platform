"""Main artefact retriever — loads any experiment artefact by key.

This is the single entry-point for reading artefacts.  It delegates to
specialised sub-retrievers for diagnostics, plots, and report manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.api.schemas import ArtefactMetadata
from src.orchestration.registry.artefact_registry import get_spec
from src.orchestration.utils.filesystem import (
    all_diagnostics_paths,
    config_path,
    equity_curve_path,
    experiment_root,
    metadata_path,
    metrics_path,
    ml_provenance_path,
    returns_path,
    weights_path,
)
from src.orchestration.utils.serialization import load_json, load_parquet


def retrieve(
    experiment_name: str,
    key: str,
    base: Path | str | None = None,
) -> Any:
    """Load artefact ``key`` for the given experiment.

    Returns the loaded object (dict / list / DataFrame) or None if the
    artefact is absent or cannot be parsed.
    """
    path = _resolve_path(experiment_name, key, base)
    if path is None or not path.exists():
        return None
    spec = get_spec(key)
    atype = spec.artefact_type if spec else _infer_type(path)
    if atype == "parquet":
        return load_parquet(path)
    if atype in ("json", "yaml"):
        return load_json(path)
    return None


def retrieve_many(
    experiment_name: str,
    keys: list[str],
    base: Path | str | None = None,
) -> dict[str, Any]:
    """Load multiple artefacts; keys with no data are omitted from result."""
    result: dict[str, Any] = {}
    for key in keys:
        data = retrieve(experiment_name, key, base)
        if data is not None:
            result[key] = data
    return result


def list_artefacts(
    experiment_name: str,
    base: Path | str | None = None,
) -> list[ArtefactMetadata]:
    """Return ArtefactMetadata for every known artefact key, marking which exist."""
    diag_paths = all_diagnostics_paths(experiment_name, base)
    experiment_root(experiment_name, base)

    core_paths: dict[str, Path] = {
        "metadata": metadata_path(experiment_name, base),
        "metrics": metrics_path(experiment_name, base),
        "config": config_path(experiment_name, base),
        "ml_provenance": ml_provenance_path(experiment_name, base),
        "equity_curve": equity_curve_path(experiment_name, base),
        "returns": returns_path(experiment_name, base),
        "weights": weights_path(experiment_name, base),
    }

    all_paths = {**core_paths, **diag_paths}
    metas: list[ArtefactMetadata] = []
    for key, path in all_paths.items():
        spec = get_spec(key)
        metas.append(ArtefactMetadata(
            key=key,
            path=str(path),
            artefact_type=spec.artefact_type if spec else _infer_type(path),
            group=spec.group if spec else "unknown",
            exists=path.exists(),
        ))
    return metas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_path(
    experiment_name: str,
    key: str,
    base: Path | str | None,
) -> Path | None:
    diag_paths = all_diagnostics_paths(experiment_name, base)
    if key in diag_paths:
        return diag_paths[key]
    core_map: dict[str, Path] = {
        "metadata": metadata_path(experiment_name, base),
        "metrics": metrics_path(experiment_name, base),
        "config": config_path(experiment_name, base),
        "ml_provenance": ml_provenance_path(experiment_name, base),
        "equity_curve": equity_curve_path(experiment_name, base),
        "returns": returns_path(experiment_name, base),
        "weights": weights_path(experiment_name, base),
    }
    return core_map.get(key)


def _infer_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {".parquet": "parquet", ".json": "json", ".yaml": "yaml", ".png": "png"}.get(
        suffix, "unknown"
    )
