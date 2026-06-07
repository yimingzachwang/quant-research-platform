"""Cosine-similarity ranking over the local semantic memory index.

Pure, deterministic vector math + metadata filters — no embeddings are computed
here (the caller embeds the query and passes the vector in), and no vector
database is used. Returns compact, provenance-aware items only: scores, paths,
hashes, tags, failure modes, and summaries — never full artefact bodies.
"""

from __future__ import annotations

import math

from src.orchestration.memory.semantic_store import SemanticRecord


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors; 0.0 for empty/zero/mismatched inputs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _passes_filters(
    rec: SemanticRecord,
    experiment_name: str,
    artefact_type: str,
    req_modes: list[str],
    req_tags: list[str],
) -> bool:
    meta = rec.metadata
    name = str(meta.get("experiment_name", "")).lower()
    if experiment_name and experiment_name not in name:
        return False
    if artefact_type and str(meta.get("artefact_type", "")).lower() != artefact_type:
        return False
    if req_modes:
        modes = {str(m).lower() for m in meta.get("failure_modes", [])}
        if not (set(req_modes) & modes):
            return False
    if req_tags:
        tags = {str(t).lower() for t in meta.get("tags", [])}
        if not (set(req_tags) & tags):
            return False
    return True


def rank_by_vector(
    records: list[SemanticRecord],
    query_vector: list[float],
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    tags: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Return up to ``top_k`` compact items ranked by cosine similarity.

    Hard metadata filters (applied as AND when provided):
      * experiment_name — record's experiment_name contains it (case-insensitive),
        so a base name matches its versioned descendants;
      * artefact_type   — exact case-insensitive match;
      * failure_modes   — record shares at least one of the requested modes;
      * tags            — record shares at least one of the requested tags.

    Ranking: cosine similarity (desc), ties broken by memory_id for determinism.
    Scores are rounded to 3 decimals. Returns compact items — never full bodies.
    """
    top_k = max(0, int(top_k)) if top_k is not None else 5
    exp_q = (experiment_name or "").strip().lower()
    type_q = (artefact_type or "").strip().lower()
    req_modes = [m.strip().lower() for m in (failure_modes or []) if m and m.strip()]
    req_tags = [t.strip().lower() for t in (tags or []) if t and t.strip()]

    scored: list[tuple[float, str, dict]] = []
    for rec in records:
        if not _passes_filters(rec, exp_q, type_q, req_modes, req_tags):
            continue
        score = cosine_similarity(query_vector, rec.embedding)
        scored.append((score, rec.memory_id, _compact_item(rec, score)))

    # Sort: score desc, then memory_id asc for stable determinism.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored[:top_k]]


def _compact_item(rec: SemanticRecord, score: float) -> dict:
    meta = rec.metadata
    return {
        "memory_id": rec.memory_id,
        "score": round(float(score), 3),
        "experiment_name": meta.get("experiment_name", ""),
        "artefact_type": meta.get("artefact_type", ""),
        "path": meta.get("path", ""),
        "context_hash": meta.get("context_hash", ""),
        "failure_modes": list(meta.get("failure_modes", [])),
        "tags": list(meta.get("tags", [])),
        "short_summary": meta.get("short_summary", ""),
    }
