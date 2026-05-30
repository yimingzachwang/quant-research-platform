"""Registry-backed dataset loading API."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from src.data.manifest import DatasetManifest, DatasetQuery
from src.data.registry import DatasetRegistry


class DatasetNotFoundError(LookupError):
    """Raised when no registered dataset matches a load request."""


class MultipleDatasetsMatchedError(LookupError):
    """Raised when a query resolves to more than one compatible dataset."""


class DatasetSchemaVersionError(ValueError):
    """Raised when matching datasets exist but schema versions are incompatible."""


class DatasetMetadataError(ValueError):
    """Raised when registry metadata is insufficient for loading."""


class DatasetColumnError(ValueError):
    """Raised when a loaded dataset is missing required columns."""


def load_dataset(
    query: DatasetQuery | None = None,
    *,
    symbol: str | None = None,
    data_type: str | None = None,
    dataset_family: str | None = None,
    frequency: str | None = None,
    source: str | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    schema_version: str | None = None,
    required_columns: Sequence[str] = (),
    registry_path: Path | str = "data/external/registry/datasets.json",
) -> pd.DataFrame:
    """Load one registered dataset by exact V1 manifest fields."""

    resolved_query = _coerce_query(
        query,
        symbol=symbol,
        data_type=data_type,
        dataset_family=dataset_family,
        frequency=frequency,
        source=source,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        schema_version=schema_version,
        required_columns=required_columns,
    )
    manifest = resolve_dataset_manifest(
        resolved_query,
        registry_path=registry_path,
    )
    storage_path = _resolve_storage_path(manifest)
    if not storage_path.exists():
        msg = f"registered dataset file is missing: {storage_path}"
        raise FileNotFoundError(msg)
    frame = pd.read_parquet(storage_path, engine="pyarrow")
    _validate_loaded_columns(frame, resolved_query)
    return frame


def find_dataset_manifest(
    *,
    symbol: str | None = None,
    data_type: str | None = None,
    dataset_family: str | None = None,
    frequency: str | None = None,
    source: str | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    schema_version: str | None = None,
    required_columns: Sequence[str] = (),
    registry_path: Path | str = "data/external/registry/datasets.json",
) -> DatasetManifest:
    """Compatibility wrapper returning the single manifest matching query fields."""

    query = DatasetQuery.from_parts(
        symbol=symbol,
        data_type=data_type,
        dataset_family=dataset_family,
        frequency=frequency,
        source=source,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        schema_version=schema_version,
        required_columns=required_columns,
    )
    return resolve_dataset_manifest(query, registry_path=registry_path)


def resolve_dataset_manifest(
    query: DatasetQuery,
    *,
    registry_path: Path | str = "data/external/registry/datasets.json",
) -> DatasetManifest:
    """Resolve a query to exactly one schema-compatible manifest."""

    registry = DatasetRegistry(registry_path)
    identity_matches = registry.query_datasets(query.without_schema_version())
    if not identity_matches:
        msg = f"dataset not found in registry: {query.model_dump(exclude_none=True)}"
        raise DatasetNotFoundError(msg)

    matches = (
        [entry for entry in identity_matches if entry.schema_version == query.schema_version]
        if query.schema_version is not None
        else identity_matches
    )
    if not matches:
        available = sorted({entry.schema_version for entry in identity_matches})
        msg = (
            f"incompatible schema version for {query.model_dump(exclude_none=True)}; "
            f"available={available}"
        )
        raise DatasetSchemaVersionError(msg)
    if len(matches) > 1:
        msg = f"multiple datasets matched {query.model_dump(exclude_none=True)}"
        raise MultipleDatasetsMatchedError(msg)

    _validate_manifest_metadata(matches[0], query)
    return matches[0]


def _coerce_query(
    query: DatasetQuery | None,
    *,
    symbol: str | None,
    data_type: str | None,
    dataset_family: str | None,
    frequency: str | None,
    source: str | None,
    dataset_id: str | None,
    dataset_name: str | None,
    schema_version: str | None,
    required_columns: Sequence[str],
) -> DatasetQuery:
    if query is not None:
        return query
    return DatasetQuery.from_parts(
        symbol=symbol,
        data_type=data_type,
        dataset_family=dataset_family,
        frequency=frequency,
        source=source,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        schema_version=schema_version,
        required_columns=required_columns,
    )


def _validate_manifest_metadata(manifest: DatasetManifest, query: DatasetQuery) -> None:
    missing = [
        field
        for field in [
            "dataset_id",
            "symbol",
            "data_type",
            "source",
            "frequency",
            "schema_version",
            "storage_path",
        ]
        if not getattr(manifest, field)
    ]
    if missing:
        msg = f"dataset manifest is missing required metadata: {missing}"
        raise DatasetMetadataError(msg)
    if manifest.data_type != query.dataset_family:
        msg = f"dataset family mismatch: {manifest.data_type} != {query.dataset_family}"
        raise DatasetMetadataError(msg)


def _resolve_storage_path(manifest: DatasetManifest) -> Path:
    if not manifest.storage_path:
        msg = f"dataset manifest has no storage path: {manifest.dataset_id}"
        raise DatasetMetadataError(msg)
    return Path(manifest.storage_path)


def _validate_loaded_columns(frame: pd.DataFrame, query: DatasetQuery) -> None:
    if not query.required_columns:
        return
    missing = sorted(set(query.required_columns) - set(frame.columns))
    if missing:
        msg = f"loaded dataset is missing required columns: {missing}"
        raise DatasetColumnError(msg)
