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
from src.orchestration.memory.semantic_indexer import (
    build_semantic_index,
    build_semantic_text,
    compute_source_hash,
)
from src.orchestration.memory.semantic_retriever import (
    cosine_similarity,
    rank_by_vector,
)
from src.orchestration.memory.semantic_store import (
    SemanticRecord,
    load_manifest,
    load_semantic_records,
    semantic_index_exists,
    write_semantic_records,
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
    # Phase 2 — semantic retrieval
    "SemanticRecord",
    "build_semantic_index",
    "build_semantic_text",
    "compute_source_hash",
    "rank_by_vector",
    "cosine_similarity",
    "semantic_index_exists",
    "load_semantic_records",
    "write_semantic_records",
    "load_manifest",
]
