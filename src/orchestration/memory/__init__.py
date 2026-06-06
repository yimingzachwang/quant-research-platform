"""Research memory layer (Phase 1 — metadata/keyword RAG).

A lightweight, controlled memory layer that indexes compact summaries of existing
Zeto artefacts into a local JSONL file and retrieves prior research evidence by
deterministic keyword / metadata matching.

Phase 1 scope: NO embeddings, NO vector database, NO semantic retrieval, NO
agent framework.  Full artefacts stay on disk; memory holds only compact,
provenance-aware pointers.
"""

from src.orchestration.memory.memory_indexer import build_memory_records
from src.orchestration.memory.memory_retriever import retrieve_memory
from src.orchestration.memory.memory_schema import (
    ARTEFACT_TYPES,
    MemoryRecord,
    compute_memory_id,
)
from src.orchestration.memory.memory_store import (
    index_exists,
    load_records,
    write_records,
)

__all__ = [
    "ARTEFACT_TYPES",
    "MemoryRecord",
    "compute_memory_id",
    "build_memory_records",
    "retrieve_memory",
    "index_exists",
    "load_records",
    "write_records",
]
