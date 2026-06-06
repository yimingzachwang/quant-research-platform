"""Local JSONL persistence for research-memory records.

The store is a single newline-delimited JSON file
(``results/research_memory/memory_index.jsonl``) — one ``MemoryRecord`` per
line.  Reads are tolerant: malformed lines are skipped, a missing file yields an
empty list.  Writes are deduplicated by ``memory_id`` so a refresh never leaves
duplicate entries.

No database, no embeddings, no network — just a small local index file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.orchestration.memory.memory_schema import MemoryRecord
from src.orchestration.utils.filesystem import memory_index_path

logger = logging.getLogger(__name__)


def index_exists(memory_base: Path | str | None = None) -> bool:
    """True if the JSONL index file is present on disk."""
    return memory_index_path(memory_base).exists()


def load_records(memory_base: Path | str | None = None) -> list[MemoryRecord]:
    """Load all memory records from the JSONL index.

    Returns [] when the index does not exist.  Malformed lines are skipped (the
    index is advisory evidence, never authoritative state).
    """
    path = memory_index_path(memory_base)
    if not path.exists():
        return []
    records: list[MemoryRecord] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(MemoryRecord.from_dict(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 — tolerate partial/corrupt lines
            logger.debug("Skipping malformed memory line in %s: %s", path, exc)
    return records


def _dedupe(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Keep the last record per memory_id, preserving first-seen order."""
    by_id: dict[str, MemoryRecord] = {}
    order: list[str] = []
    for rec in records:
        if rec.memory_id not in by_id:
            order.append(rec.memory_id)
        by_id[rec.memory_id] = rec
    return [by_id[mid] for mid in order]


def write_records(
    records: list[MemoryRecord],
    memory_base: Path | str | None = None,
) -> Path:
    """Overwrite the JSONL index with ``records`` (deduplicated by memory_id).

    Returns the written path.  Creates the parent directory as needed.  This
    fully refreshes the index — it never appends silently to stale entries.
    """
    path = memory_index_path(memory_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = _dedupe(records)
    lines = [json.dumps(rec.to_dict(), sort_keys=True) for rec in deduped]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path
