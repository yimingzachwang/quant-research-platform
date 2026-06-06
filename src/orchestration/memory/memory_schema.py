"""Compact research-memory record schema (Phase 1 metadata/keyword RAG).

A ``MemoryRecord`` is a small, provenance-aware pointer to one persisted Zeto
artefact — never the artefact's full contents.  Records are stored one per line
in a local JSONL index and retrieved by deterministic keyword / metadata
matching.  No embeddings, no vector store, no LLM.

Each record carries only:
  * identity + provenance (memory_id, context_hash, path, created_at);
  * coordinates (experiment_name, session_id, artefact_type);
  * compact, searchable signal (failure_modes, tags, short_summary).

Full artefacts stay on disk; ``path`` references them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Controlled artefact-type vocabulary.  The indexer assigns exactly one of these
# to every record; the retriever filters on them.  Keep this list and the
# indexer's source map in sync.
ARTEFACT_TYPES: tuple[str, ...] = (
    "experiment_metadata",
    "experiment_metrics",
    "llm_review",
    "iteration_proposal",
    "draft",
    "report",
    "session",
)

# Hard cap so a single summary can never stream a full artefact into a response.
SUMMARY_MAX_CHARS = 240


def compute_memory_id(artefact_type: str, path: str) -> str:
    """Return a stable, deterministic id for one artefact.

    The id is derived from ``artefact_type`` + ``path`` so re-indexing the same
    artefact always yields the same id — guaranteeing no duplicates on refresh.
    """
    digest = hashlib.sha256(f"{artefact_type}:{path}".encode()).hexdigest()
    return f"mem_{digest[:16]}"


@dataclass
class MemoryRecord:
    """One compact memory item pointing at a persisted artefact."""

    memory_id: str
    experiment_name: str
    artefact_type: str
    path: str
    short_summary: str
    created_at: str = ""
    context_hash: str = ""
    session_id: str | None = None
    failure_modes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "experiment_name": self.experiment_name,
            "session_id": self.session_id,
            "artefact_type": self.artefact_type,
            "context_hash": self.context_hash,
            "path": self.path,
            "created_at": self.created_at,
            "failure_modes": list(self.failure_modes),
            "tags": list(self.tags),
            "short_summary": self.short_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        return cls(
            memory_id=data["memory_id"],
            experiment_name=data.get("experiment_name", ""),
            artefact_type=data.get("artefact_type", ""),
            path=data.get("path", ""),
            short_summary=data.get("short_summary", ""),
            created_at=data.get("created_at", ""),
            context_hash=data.get("context_hash", "") or "",
            session_id=data.get("session_id"),
            failure_modes=list(data.get("failure_modes", []) or []),
            tags=list(data.get("tags", []) or []),
        )
