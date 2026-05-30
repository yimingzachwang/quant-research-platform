"""JSON-backed dataset registry."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.data.models import DatasetMetadata


class JsonDatasetRegistry:
    """Maintain dataset metadata entries in a single JSON index."""

    def __init__(self, project_root: Path | str) -> None:
        self.registry_root = Path(project_root) / "data" / "external" / "registry"
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_root / "datasets.json"

    def upsert(self, metadata: DatasetMetadata) -> None:
        """Create or replace one registry entry."""

        entries = self.list_entries()
        entries[metadata.dataset_id] = metadata
        payload = {
            dataset_id: entry.model_dump(mode="json")
            for dataset_id, entry in sorted(entries.items(), key=lambda item: item[0])
        }
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        logger.bind(dataset_id=metadata.dataset_id, path=str(self.index_path)).info(
            "updated dataset registry"
        )

    def get(self, dataset_id: str) -> DatasetMetadata | None:
        """Return metadata for a dataset id if present."""

        return self.list_entries().get(dataset_id)

    def list_entries(self) -> dict[str, DatasetMetadata]:
        """Return all registry entries keyed by dataset id."""

        if not self.index_path.exists():
            return {}
        payload = json.loads(self.index_path.read_text())
        return {
            dataset_id: DatasetMetadata.model_validate(entry)
            for dataset_id, entry in payload.items()
            if not dataset_id.startswith("_")
        }
