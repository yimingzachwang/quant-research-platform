"""Deterministic keyword / metadata retrieval over research-memory records.

Phase 1 RAG: exact, case-insensitive keyword and metadata matching only — no
embeddings, no vector store, no semantic similarity.  Retrieval applies hard
filters (experiment_name, artefact_type, failure_modes), then ranks the survivors
by a simple additive score over matched fields, and returns the top_k as compact
dicts (summaries, paths, hashes, tags, matched terms — never full artefacts).
"""

from __future__ import annotations

import re

from src.orchestration.memory.memory_schema import MemoryRecord

# Scoring weights — failure-mode and metadata matches outrank loose keyword hits.
_W_FAILURE_MODE = 3
_W_EXPERIMENT = 2
_W_ARTEFACT_TYPE = 2
_W_QUERY_TERM = 1

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _searchable_text(rec: MemoryRecord) -> str:
    return " ".join([
        rec.experiment_name,
        rec.artefact_type,
        rec.short_summary,
        " ".join(rec.tags),
        " ".join(rec.failure_modes),
    ]).lower()


def retrieve_memory(
    records: list[MemoryRecord],
    query: str | None = None,
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Return up to ``top_k`` compact memory items matching the criteria.

    Filters (applied as AND when provided):
      * experiment_name — record's experiment_name contains it (case-insensitive),
        so a base name matches its versioned descendants;
      * artefact_type   — exact case-insensitive match;
      * failure_modes   — record shares at least one of the requested modes.

    Ranking: additive score over matched failure modes, experiment/type match,
    and query keyword hits; ties broken by recency (created_at) then memory_id.
    When any criterion is supplied, only records with a positive score are
    returned; with no criteria at all, the most recent ``top_k`` are returned.
    """
    top_k = max(0, int(top_k)) if top_k is not None else 5
    exp_q = (experiment_name or "").strip().lower()
    type_q = (artefact_type or "").strip().lower()
    req_modes = [m.strip().lower() for m in (failure_modes or []) if m and m.strip()]
    query_tokens = _tokenize(query or "")
    has_criteria = bool(exp_q or type_q or req_modes or query_tokens)

    scored: list[tuple[int, str, str, dict]] = []
    for rec in records:
        # --- hard filters ---------------------------------------------------
        if exp_q and exp_q not in rec.experiment_name.lower():
            continue
        if type_q and rec.artefact_type.lower() != type_q:
            continue
        rec_modes_lower = {m.lower() for m in rec.failure_modes}
        if req_modes and not (set(req_modes) & rec_modes_lower):
            continue

        # --- scoring + matched terms ---------------------------------------
        score = 0
        matched: list[str] = []

        for mode in req_modes:
            if mode in rec_modes_lower:
                score += _W_FAILURE_MODE
                # Echo the record's original-cased failure mode name.
                matched.append(_original_mode(rec, mode))

        if exp_q and exp_q in rec.experiment_name.lower():
            score += _W_EXPERIMENT
            matched.append(rec.experiment_name)

        if type_q and rec.artefact_type.lower() == type_q:
            score += _W_ARTEFACT_TYPE
            matched.append(rec.artefact_type)

        if query_tokens:
            haystack = set(_tokenize(_searchable_text(rec)))
            for tok in query_tokens:
                if tok in haystack and tok not in [m.lower() for m in matched]:
                    score += _W_QUERY_TERM
                    matched.append(tok)

        if has_criteria and score <= 0:
            continue

        scored.append((score, rec.created_at or "", rec.memory_id, _compact_item(rec, matched)))

    # Sort: score desc, then recency desc, then id for determinism.
    scored.sort(key=lambda t: (-t[0], _neg_str(t[1]), t[2]))
    return [item for _, _, _, item in scored[:top_k]]


def _original_mode(rec: MemoryRecord, lowered: str) -> str:
    for m in rec.failure_modes:
        if m.lower() == lowered:
            return m
    return lowered


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _compact_item(rec: MemoryRecord, matched: list[str]) -> dict:
    """Compact, provenance-aware item — never the full artefact body."""
    return {
        "memory_id": rec.memory_id,
        "experiment_name": rec.experiment_name,
        "artefact_type": rec.artefact_type,
        "path": rec.path,
        "context_hash": rec.context_hash,
        "failure_modes": list(rec.failure_modes),
        "tags": list(rec.tags),
        "matched_terms": _dedupe_preserve(matched),
        "short_summary": rec.short_summary,
    }


def _neg_str(s: str) -> tuple:
    """Sort key that orders strings descending (most recent created_at first)."""
    # Invert each codepoint so a plain ascending sort yields descending order.
    return tuple(-ord(c) for c in s)
