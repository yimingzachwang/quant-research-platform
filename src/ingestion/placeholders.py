"""No-op ingestion implementations for architecture tests."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.interfaces import IngestionResult


@dataclass(frozen=True)
class NoOpIngestionJob:
    """Ingestion job that records intent without touching external systems."""

    dataset: str

    def run(self) -> IngestionResult:
        """Return an empty ingestion result."""
        return IngestionResult(dataset=self.dataset)
