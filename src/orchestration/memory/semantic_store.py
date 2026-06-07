"""Local persistence for the semantic (embedding) research-memory index.

Phase 2 keeps the store deliberately simple and transparent — no vector
database. Embeddings live one-per-line in a JSONL file alongside a small
manifest describing the embedding model and dimension:

    results/research_memory/semantic_memory_index.jsonl
    results/research_memory/semantic_memory_manifest.json

Reads are tolerant (missing file -> empty; malformed line -> skipped). Writes
deduplicate by ``memory_id`` so a refresh never leaves duplicate embeddings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.orchestration.utils.filesystem import (
    semantic_memory_index_path,
    semantic_memory_manifest_path,
)
from src.orchestration.utils.serialization import dump_json, load_json

logger = logging.getLogger(__name__)


@dataclass
class SemanticRecord:
    """One embedded memory item pointing at a Phase 1 memory record."""

    memory_id: str
    embedding_model: str
    embedding_dim: int
    embedding: list[float]
    source_hash: str
    indexed_at: str
    text_preview: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding": list(self.embedding),
            "source_hash": self.source_hash,
            "indexed_at": self.indexed_at,
            "text_preview": self.text_preview,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticRecord:
        return cls(
            memory_id=data["memory_id"],
            embedding_model=data.get("embedding_model", ""),
            embedding_dim=int(data.get("embedding_dim", 0) or 0),
            embedding=list(data.get("embedding", []) or []),
            source_hash=data.get("source_hash", "") or "",
            indexed_at=data.get("indexed_at", "") or "",
            text_preview=data.get("text_preview", "") or "",
            metadata=dict(data.get("metadata", {}) or {}),
        )


def semantic_index_exists(memory_base: Path | str | None = None) -> bool:
    """True if the semantic JSONL index file is present on disk."""
    return semantic_memory_index_path(memory_base).exists()


def load_semantic_records(memory_base: Path | str | None = None) -> list[SemanticRecord]:
    """Load all semantic records from the JSONL index.

    Returns [] when the index is missing. Malformed lines are skipped (the
    index is advisory evidence, never authoritative state).
    """
    path = semantic_memory_index_path(memory_base)
    if not path.exists():
        return []
    records: list[SemanticRecord] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(SemanticRecord.from_dict(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 — tolerate partial/corrupt lines
            logger.debug("Skipping malformed semantic line in %s: %s", path, exc)
    return records


def load_manifest(memory_base: Path | str | None = None) -> dict[str, Any] | None:
    """Load the semantic-memory manifest, or None if absent/malformed."""
    data = load_json(semantic_memory_manifest_path(memory_base))
    return data if isinstance(data, dict) else None


def _dedupe(records: list[SemanticRecord]) -> list[SemanticRecord]:
    """Keep the last record per memory_id, preserving first-seen order."""
    by_id: dict[str, SemanticRecord] = {}
    order: list[str] = []
    for rec in records:
        if rec.memory_id not in by_id:
            order.append(rec.memory_id)
        by_id[rec.memory_id] = rec
    return [by_id[mid] for mid in order]


def write_semantic_records(
    records: list[SemanticRecord],
    manifest: dict[str, Any],
    memory_base: Path | str | None = None,
) -> Path:
    """Overwrite the semantic JSONL index (deduped by memory_id) + manifest.

    Returns the written index path. Fully refreshes the index — it never appends
    silently to stale entries.
    """
    index_path = semantic_memory_index_path(memory_base)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    deduped = _dedupe(records)
    lines = [json.dumps(rec.to_dict(), sort_keys=True) for rec in deduped]
    index_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    dump_json(manifest, semantic_memory_manifest_path(memory_base))
    return index_path
