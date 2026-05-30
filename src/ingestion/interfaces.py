"""Data ingestion contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class IngestionResult:
    """Result metadata for a dataset ingestion job."""

    dataset: str
    rows_ingested: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class DataIngestionJob(Protocol):
    """Loads raw vendor data into a validated research dataset."""

    def run(self) -> IngestionResult:
        """Execute ingestion and return dataset metadata."""
        raise NotImplementedError
