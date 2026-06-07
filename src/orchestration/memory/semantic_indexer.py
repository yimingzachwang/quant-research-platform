"""Build a local semantic (embedding) index from Phase 1 memory records.

Phase 2 layers semantic retrieval on top of Phase 1 WITHOUT touching raw files:
the source of truth is the Phase 1 JSONL index
(``results/research_memory/memory_index.jsonl``). Each compact ``MemoryRecord``
becomes one short text, which is embedded with a local OpenAI-compatible
embedding model (e.g. LM Studio's nomic model).

Constraints honoured here:
  * embeds compact summaries ONLY — never raw reports, data, parquet, or plots;
  * calls the embeddings endpoint ONLY — never a chat/completion model;
  * runs no experiment, approves nothing, renders nothing, executes nothing;
  * re-indexing refreshes in place (stable memory_id) and reuses unchanged
    embeddings via a source hash, so it never duplicates or needlessly re-embeds.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.orchestration.llm.llm_interface import DEFAULT_EMBEDDING_MODEL, embed_texts
from src.orchestration.memory.memory_schema import SUMMARY_MAX_CHARS, MemoryRecord
from src.orchestration.memory.memory_store import load_records as load_memory_records
from src.orchestration.memory.semantic_store import (
    SemanticRecord,
    load_semantic_records,
    write_semantic_records,
)
from src.orchestration.utils.filesystem import semantic_memory_index_path

logger = logging.getLogger(__name__)

# Preview cap so a record never carries more than a compact line of text.
_PREVIEW_MAX_CHARS = SUMMARY_MAX_CHARS


def build_semantic_text(record: MemoryRecord) -> str:
    """Compose the compact text to embed from a Phase 1 memory record.

    Uses only the record's compact fields — never the underlying artefact body.
    """
    return "\n".join([
        f"Experiment: {record.experiment_name}",
        f"Artefact type: {record.artefact_type}",
        f"Failure modes: {', '.join(record.failure_modes) or 'none'}",
        f"Tags: {', '.join(record.tags) or 'none'}",
        f"Summary: {record.short_summary}",
        f"Path: {record.path}",
        f"Context hash: {record.context_hash}",
    ])


def compute_source_hash(text: str) -> str:
    """Deterministic hash of the embed text — used to skip unchanged re-embeds."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _metadata(record: MemoryRecord) -> dict:
    return {
        "experiment_name": record.experiment_name,
        "artefact_type": record.artefact_type,
        "context_hash": record.context_hash,
        "path": record.path,
        "failure_modes": list(record.failure_modes),
        "tags": list(record.tags),
        "short_summary": record.short_summary,
    }


def build_semantic_index(
    provider: str = "openai",
    model: str | None = None,
    base_url: str | None = None,
    memory_base: Path | str | None = None,
    embed_fn=None,
) -> dict:
    """Embed all Phase 1 memory records into the local semantic index.

    Reuses existing embeddings whose source text and model are unchanged; embeds
    only new/changed records. Refreshes in place: the resulting index contains
    exactly the current Phase 1 memory_ids (stale entries are dropped).

    Returns a status dict:
        {status, embedded_count, newly_embedded, reused, embedding_model,
         embedding_dim, index_path}
      status is "ok", "no_phase1_index", or "embedding_failed".
    """
    # Resolve at call time (not as a default arg) so tests can patch embed_texts.
    if embed_fn is None:
        embed_fn = embed_texts
    resolved_model = model or DEFAULT_EMBEDDING_MODEL
    index_path = str(semantic_memory_index_path(memory_base))

    records = load_memory_records(memory_base)
    if not records:
        return {
            "status": "no_phase1_index",
            "embedded_count": 0,
            "newly_embedded": 0,
            "reused": 0,
            "embedding_model": resolved_model,
            "embedding_dim": 0,
            "index_path": index_path,
        }

    existing = {r.memory_id: r for r in load_semantic_records(memory_base)}

    plan: list[tuple[MemoryRecord, str, str]] = []  # (record, text, source_hash)
    reuse: dict[str, SemanticRecord] = {}
    to_embed_ids: list[str] = []
    to_embed_texts: list[str] = []
    for rec in records:
        text = build_semantic_text(rec)
        source_hash = compute_source_hash(text)
        prev = existing.get(rec.memory_id)
        if (
            prev is not None
            and prev.source_hash == source_hash
            and prev.embedding_model == resolved_model
            and prev.embedding
        ):
            reuse[rec.memory_id] = prev
        else:
            to_embed_ids.append(rec.memory_id)
            to_embed_texts.append(text)
        plan.append((rec, text, source_hash))

    new_vectors: dict[str, list[float]] = {}
    if to_embed_texts:
        try:
            resp = embed_fn(
                to_embed_texts, provider=provider, model=resolved_model, base_url=base_url
            )
        except Exception as exc:  # noqa: BLE001 — clean envelope, never a traceback
            return {
                "status": "embedding_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "embedded_count": 0,
                "newly_embedded": 0,
                "reused": len(reuse),
                "embedding_model": resolved_model,
                "embedding_dim": 0,
                "index_path": index_path,
            }
        resolved_model = resp.model or resolved_model
        for mid, vec in zip(to_embed_ids, resp.vectors):
            new_vectors[mid] = list(vec)

    indexed_at = datetime.now(UTC).isoformat()
    out: list[SemanticRecord] = []
    for rec, text, source_hash in plan:
        embedding = new_vectors.get(rec.memory_id)
        if embedding is None:
            embedding = reuse[rec.memory_id].embedding
        out.append(SemanticRecord(
            memory_id=rec.memory_id,
            embedding_model=resolved_model,
            embedding_dim=len(embedding),
            embedding=embedding,
            source_hash=source_hash,
            indexed_at=indexed_at,
            text_preview=_truncate(text),
            metadata=_metadata(rec),
        ))

    embedding_dim = out[0].embedding_dim if out else 0
    manifest = {
        "embedding_model": resolved_model,
        "embedding_dim": embedding_dim,
        "item_count": len(out),
        "indexed_at": indexed_at,
        "source_index": "results/research_memory/memory_index.jsonl",
    }
    write_semantic_records(out, manifest, memory_base)

    return {
        "status": "ok",
        "embedded_count": len(out),
        "newly_embedded": len(new_vectors),
        "reused": len(reuse),
        "embedding_model": resolved_model,
        "embedding_dim": embedding_dim,
        "index_path": index_path,
    }


def _truncate(text: str, limit: int = _PREVIEW_MAX_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
