"""Simple JSON-backed manifest registry for ingestion runs."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.manifest import DatasetManifest, DatasetQuery


class DatasetRegistry:
    """Append-safe deterministic registry for dataset manifests."""

    manifest_key = "_manifests"

    def __init__(self, registry_path: Path | str = "data/external/registry/datasets.json") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("{}\n")

    def register(self, manifest: DatasetManifest) -> None:
        """Register one manifest, replacing any existing matching request hash."""

        entries = [
            entry for entry in self.load_all() if entry.request_hash != manifest.request_hash
        ]
        entries.append(manifest)
        self._write_entries(entries)

    def exists(self, request_hash: str) -> bool:
        """Return whether a request hash is already registered."""

        return any(entry.request_hash == request_hash for entry in self.load_all())

    def load_all(self) -> list[DatasetManifest]:
        """Load all registry manifests."""

        if not self.registry_path.exists():
            self.registry_path.write_text("{}\n")
            return []
        payload = json.loads(self.registry_path.read_text())
        if not isinstance(payload, dict):
            return []
        manifests = payload.get(self.manifest_key, [])
        return [DatasetManifest.model_validate(entry) for entry in manifests]

    def find(
        self,
        *,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        symbol: str | None = None,
        source: str | None = None,
        data_type: str | None = None,
        frequency: str | None = None,
        request_hash: str | None = None,
    ) -> list[DatasetManifest]:
        """Find manifests by a small set of V1 metadata fields."""

        entries = self.load_all()
        if dataset_id is not None:
            entries = [entry for entry in entries if entry.dataset_id == dataset_id]
        if dataset_name is not None:
            entries = [entry for entry in entries if entry.dataset_name == dataset_name]
        if symbol is not None:
            entries = [entry for entry in entries if entry.symbol == symbol.upper()]
        if source is not None:
            entries = [entry for entry in entries if entry.source == source]
        if data_type is not None:
            entries = [entry for entry in entries if entry.data_type == data_type]
        if frequency is not None:
            entries = [entry for entry in entries if entry.frequency == frequency]
        if request_hash is not None:
            entries = [entry for entry in entries if entry.request_hash == request_hash]
        return entries

    def query_datasets(self, query: DatasetQuery) -> list[DatasetManifest]:
        """Find manifests matching a typed dataset query."""

        entries = self.load_all()
        if query.dataset_id is not None:
            entries = [entry for entry in entries if entry.dataset_id == query.dataset_id]
        if query.dataset_name is not None:
            entries = [entry for entry in entries if entry.dataset_name == query.dataset_name]
        if query.dataset_family is not None:
            entries = [entry for entry in entries if entry.data_type == query.dataset_family]
        if query.symbol is not None:
            entries = [entry for entry in entries if entry.symbol == query.symbol]
        if query.source is not None:
            entries = [entry for entry in entries if entry.source == query.source]
        if query.frequency is not None:
            entries = [entry for entry in entries if entry.frequency == query.frequency]
        if query.schema_version is not None:
            entries = [entry for entry in entries if entry.schema_version == query.schema_version]
        return entries

    def find_by_symbol(self, symbol: str) -> list[DatasetManifest]:
        """Find manifests for an exact symbol."""

        return self.find(symbol=symbol)

    def find_by_source(self, source: str) -> list[DatasetManifest]:
        """Find manifests for an exact source."""

        return self.find(source=source)

    def _write_entries(self, entries: list[DatasetManifest]) -> None:
        ordered = sorted(
            entries,
            key=lambda entry: (
                entry.dataset_name,
                entry.symbol,
                entry.data_type,
                entry.source,
                entry.frequency,
                entry.start_date,
                entry.end_date,
                entry.request_hash,
            ),
        )
        if self.registry_path.exists():
            payload = json.loads(self.registry_path.read_text())
            if not isinstance(payload, dict):
                payload = {}
        else:
            payload = {}
        payload[self.manifest_key] = [entry.model_dump(mode="json") for entry in ordered]
        temporary_path = self.registry_path.with_suffix(f"{self.registry_path.suffix}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(self.registry_path)
