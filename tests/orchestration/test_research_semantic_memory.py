"""Tests for Phase 2 Semantic Research Memory (local embedding retrieval).

All tests use tmp_path for isolation and either the deterministic ``stub``
embedding provider or a mocked embedding function — LM Studio is never required.
Phase 2 adds local cosine-similarity retrieval over the Phase 1 records: no
vector database, no chat/completion LLM, no raw-file access.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from unittest.mock import patch

from src.orchestration.api import research_api as api
from src.orchestration.memory import (
    memory_store,
    semantic_indexer,
    semantic_retriever,
    semantic_store,
)
from src.orchestration.memory.memory_schema import MemoryRecord, compute_memory_id
from src.orchestration.memory.semantic_store import SemanticRecord

_MEMBASE = "results" + "/research_memory"  # joined via tmp_path below

_RA = "src.orchestration.api.research_api"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _record(name, artefact_type, summary, failure_modes=None, tags=None) -> MemoryRecord:
    path = f"results/llm_reviews/{name}/{artefact_type}.json"
    return MemoryRecord(
        memory_id=compute_memory_id(artefact_type, path),
        experiment_name=name,
        artefact_type=artefact_type,
        path=path,
        short_summary=summary,
        created_at="2026-06-02T00:00:00+00:00",
        context_hash="c" * 64,
        session_id=None,
        failure_modes=failure_modes or [],
        tags=tags or [],
    )


def _phase1_records() -> list[MemoryRecord]:
    return [
        _record(
            "canonical_ml_showcase", "llm_review",
            "Post-run review found momentum instability and poor out-of-sample consistency.",
            failure_modes=["poor_oos_consistency", "catastrophic_split"],
            tags=["validation", "momentum", "oos_consistency"],
        ),
        _record(
            "canonical_ml_showcase_v2", "iteration_proposal",
            "Proposal to increase ridge alpha and remove mom_60 feature.",
            failure_modes=[],
            tags=["regularisation", "features"],
        ),
        _record(
            "other_strategy", "experiment_metrics",
            "Metrics: sharpe=1.21, max_drawdown=-0.12.",
            failure_modes=[],
            tags=["sharpe"],
        ),
    ]


def _write_phase1(tmp_path: Path, records=None) -> Path:
    memory_base = tmp_path / "results" / "research_memory"
    memory_store.write_records(records or _phase1_records(), memory_base)
    return memory_base


# ---------------------------------------------------------------------------
# 1. Semantic text construction
# ---------------------------------------------------------------------------


class TestSemanticText:
    def test_build_semantic_text_contains_compact_fields(self):
        rec = _phase1_records()[0]
        text = semantic_indexer.build_semantic_text(rec)
        assert "Experiment: canonical_ml_showcase" in text
        assert "Artefact type: llm_review" in text
        assert "Failure modes: poor_oos_consistency, catastrophic_split" in text
        assert "Tags: validation, momentum, oos_consistency" in text
        assert "Summary:" in text and "momentum instability" in text
        assert "Path: results/llm_reviews/canonical_ml_showcase/llm_review.json" in text
        assert "Context hash:" in text

    def test_source_hash_is_deterministic_and_changes_with_text(self):
        rec = _phase1_records()[0]
        h1 = semantic_indexer.compute_source_hash(semantic_indexer.build_semantic_text(rec))
        h2 = semantic_indexer.compute_source_hash(semantic_indexer.build_semantic_text(rec))
        assert h1 == h2
        assert semantic_indexer.compute_source_hash("different") != h1


# ---------------------------------------------------------------------------
# 2. Store roundtrip
# ---------------------------------------------------------------------------


class TestStoreRoundtrip:
    def test_write_read_roundtrip(self, tmp_path):
        memory_base = tmp_path / "results" / "research_memory"
        rec = SemanticRecord(
            memory_id="mem_1",
            embedding_model="text-embedding-nomic-embed-text-v1.5",
            embedding_dim=3,
            embedding=[0.1, 0.2, 0.3],
            source_hash="abc123",
            indexed_at="2026-06-06T00:00:00+00:00",
            text_preview="preview",
            metadata={"experiment_name": "e", "artefact_type": "llm_review",
                      "path": "p", "context_hash": "h", "failure_modes": [], "tags": [],
                      "short_summary": "s"},
        )
        manifest = {"embedding_model": rec.embedding_model, "embedding_dim": 3,
                    "item_count": 1, "indexed_at": rec.indexed_at}
        semantic_store.write_semantic_records([rec], manifest, memory_base)

        loaded = semantic_store.load_semantic_records(memory_base)
        assert len(loaded) == 1
        assert loaded[0].to_dict() == rec.to_dict()
        man = semantic_store.load_manifest(memory_base)
        assert man["embedding_model"] == rec.embedding_model
        assert man["embedding_dim"] == 3

    def test_missing_index_returns_empty(self, tmp_path):
        memory_base = tmp_path / "results" / "research_memory"
        assert semantic_store.load_semantic_records(memory_base) == []
        assert semantic_store.semantic_index_exists(memory_base) is False
        assert semantic_store.load_manifest(memory_base) is None


# ---------------------------------------------------------------------------
# 3. Indexing + re-index dedup (stub embeddings — deterministic, offline)
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_index_embeds_all_phase1_records(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        result = api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        assert result["status"] == "ok"
        assert result["embedded_count"] == 3
        recs = semantic_store.load_semantic_records(memory_base)
        assert len(recs) == 3
        assert {r.memory_id for r in recs} == {r.memory_id for r in _phase1_records()}
        assert all(r.embedding_dim == len(r.embedding) > 0 for r in recs)

    def test_no_duplicate_embeddings_on_reindex(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        first = api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        second = api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        recs = semantic_store.load_semantic_records(memory_base)
        ids = [r.memory_id for r in recs]
        assert first["embedded_count"] == second["embedded_count"] == 3
        assert len(ids) == len(set(ids))
        # Second pass reuses unchanged embeddings — nothing re-embedded.
        assert second["newly_embedded"] == 0
        assert second["reused"] == 3

    def test_reindex_reembeds_only_changed_records(self, tmp_path):
        recs = _phase1_records()
        memory_base = _write_phase1(tmp_path, recs)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        # Change one record's summary and rewrite the Phase 1 index.
        recs[0].short_summary = "Completely different summary about turnover."
        memory_store.write_records(recs, memory_base)
        result = api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        assert result["newly_embedded"] == 1
        assert result["reused"] == 2

    def test_reindex_drops_stale_semantic_records(self, tmp_path):
        recs = _phase1_records()
        memory_base = _write_phase1(tmp_path, recs)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        # Remove one Phase 1 record, re-index: stale embedding must be dropped.
        memory_store.write_records(recs[:2], memory_base)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        sem = semantic_store.load_semantic_records(memory_base)
        assert {r.memory_id for r in sem} == {r.memory_id for r in recs[:2]}

    def test_index_no_phase1_returns_status(self, tmp_path):
        memory_base = tmp_path / "results" / "research_memory"
        result = api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        assert result["status"] == "no_phase1_index"
        assert result["embedded_count"] == 0

    def test_index_embedding_failure_returns_clean_status(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        with patch.object(
            semantic_indexer, "embed_texts", side_effect=RuntimeError("endpoint down")
        ):
            result = api.index_semantic_research_memory(
                provider="openai", memory_base=memory_base
            )
        assert result["status"] == "embedding_failed"
        assert "endpoint down" in result["error"]
        # Nothing was written.
        assert semantic_store.load_semantic_records(memory_base) == []


# ---------------------------------------------------------------------------
# 4. Cosine similarity ranking (pure)
# ---------------------------------------------------------------------------


class TestCosineRanking:
    def test_cosine_identical_and_orthogonal(self):
        assert semantic_retriever.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
        assert semantic_retriever.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
        assert semantic_retriever.cosine_similarity([], [1.0]) == 0.0

    def _sem(self, mid, vec, **meta) -> SemanticRecord:
        base = {"experiment_name": mid, "artefact_type": "llm_review", "path": f"/{mid}",
                "context_hash": "h", "failure_modes": [], "tags": [], "short_summary": mid}
        base.update(meta)
        return SemanticRecord(
            memory_id=mid, embedding_model="m", embedding_dim=len(vec), embedding=vec,
            source_hash="s", indexed_at="t", text_preview=mid, metadata=base,
        )

    def test_rank_orders_by_similarity_with_rounded_scores(self):
        recs = [
            self._sem("near", [1.0, 0.0, 0.0]),
            self._sem("mid", [0.7, 0.7, 0.0]),
            self._sem("far", [0.0, 0.0, 1.0]),
        ]
        items = semantic_retriever.rank_by_vector(recs, [1.0, 0.0, 0.0], top_k=3)
        assert [it["memory_id"] for it in items] == ["near", "mid", "far"]
        assert items[0]["score"] == 1.0
        assert items[2]["score"] == 0.0
        # Scores rounded to 3 decimals.
        assert all(round(it["score"], 3) == it["score"] for it in items)

    def test_top_k_limits_results(self):
        recs = [self._sem(f"r{i}", [1.0, 0.0]) for i in range(5)]
        items = semantic_retriever.rank_by_vector(recs, [1.0, 0.0], top_k=2)
        assert len(items) == 2


# ---------------------------------------------------------------------------
# 6-8. Metadata filters (pure)
# ---------------------------------------------------------------------------


class TestFilters:
    def _records(self):
        return [
            SemanticRecord("a", "m", 2, [1.0, 0.0], "s", "t", "a",
                           {"experiment_name": "canonical_ml_showcase", "artefact_type": "llm_review",
                            "path": "/a", "context_hash": "h",
                            "failure_modes": ["poor_oos_consistency"], "tags": ["momentum"],
                            "short_summary": "a"}),
            SemanticRecord("b", "m", 2, [1.0, 0.0], "s", "t", "b",
                           {"experiment_name": "canonical_ml_showcase_v2", "artefact_type": "iteration_proposal",
                            "path": "/b", "context_hash": "h",
                            "failure_modes": [], "tags": ["regularisation"],
                            "short_summary": "b"}),
            SemanticRecord("c", "m", 2, [1.0, 0.0], "s", "t", "c",
                           {"experiment_name": "other", "artefact_type": "llm_review",
                            "path": "/c", "context_hash": "h",
                            "failure_modes": ["catastrophic_split"], "tags": ["sharpe"],
                            "short_summary": "c"}),
        ]

    def test_filter_experiment_name_includes_versions(self):
        items = semantic_retriever.rank_by_vector(
            self._records(), [1.0, 0.0], experiment_name="canonical_ml_showcase", top_k=10
        )
        names = {it["experiment_name"] for it in items}
        assert names == {"canonical_ml_showcase", "canonical_ml_showcase_v2"}

    def test_filter_failure_modes(self):
        items = semantic_retriever.rank_by_vector(
            self._records(), [1.0, 0.0], failure_modes=["poor_oos_consistency"], top_k=10
        )
        assert [it["memory_id"] for it in items] == ["a"]

    def test_filter_artefact_type_case_insensitive(self):
        items = semantic_retriever.rank_by_vector(
            self._records(), [1.0, 0.0], artefact_type="ITERATION_PROPOSAL", top_k=10
        )
        assert [it["memory_id"] for it in items] == ["b"]

    def test_filter_tags(self):
        items = semantic_retriever.rank_by_vector(
            self._records(), [1.0, 0.0], tags=["sharpe"], top_k=10
        )
        assert [it["memory_id"] for it in items] == ["c"]


# ---------------------------------------------------------------------------
# 5, 9, 10, 11. Retrieval via Research API
# ---------------------------------------------------------------------------


class TestSemanticRetrieveApi:
    def test_semantic_query_returns_related(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        # Stub embeddings are token-hash bags: a query sharing tokens with the
        # first record's text ranks it highly.
        result = api.semantic_retrieve_research_memory(
            query="momentum instability and poor out-of-sample consistency",
            provider="stub", memory_base=memory_base, top_k=3,
        )
        assert result["status"] == "ok"
        assert result["items"]
        assert result["items"][0]["experiment_name"] == "canonical_ml_showcase"
        assert "score" in result["items"][0]

    def test_retrieve_with_failure_mode_filter(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        result = api.semantic_retrieve_research_memory(
            query="instability", provider="stub", memory_base=memory_base,
            failure_modes=["poor_oos_consistency"], top_k=5,
        )
        assert result["status"] == "ok"
        for it in result["items"]:
            assert "poor_oos_consistency" in it["failure_modes"]

    def test_retrieve_no_phase1_index(self, tmp_path):
        memory_base = tmp_path / "results" / "research_memory"
        result = api.semantic_retrieve_research_memory(
            query="x", provider="stub", memory_base=memory_base
        )
        assert result["status"] == "no_phase1_index"
        assert result["items"] == []

    def test_retrieve_no_semantic_index(self, tmp_path):
        memory_base = _write_phase1(tmp_path)  # Phase 1 exists, semantic does not
        result = api.semantic_retrieve_research_memory(
            query="x", provider="stub", memory_base=memory_base
        )
        assert result["status"] == "no_semantic_index"
        assert result["items"] == []

    def test_retrieve_embedding_failure_clean(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        with patch(f"{_RA}.embed_texts", side_effect=RuntimeError("endpoint down")):
            result = api.semantic_retrieve_research_memory(
                query="x", provider="openai", memory_base=memory_base
            )
        assert result["status"] == "embedding_failed"
        assert result["items"] == []
        assert "endpoint down" in result["error"]

    def test_status_after_index(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        status = api.get_semantic_research_memory_status(memory_base=memory_base)
        assert status["index_exists"] is True
        assert status["item_count"] == 3
        assert status["embedding_dim"] > 0

    def test_status_when_missing(self, tmp_path):
        memory_base = tmp_path / "results" / "research_memory"
        status = api.get_semantic_research_memory_status(memory_base=memory_base)
        assert status["index_exists"] is False
        assert status["item_count"] == 0


# ---------------------------------------------------------------------------
# 14-16. Phase-2 constraints: no chat LLM / vector DB / execution / mutation
# ---------------------------------------------------------------------------


class TestPhase2Constraints:
    def test_no_chat_llm_or_vectordb_imports(self):
        for mod in (semantic_indexer, semantic_retriever, semantic_store):
            src = inspect.getsource(mod)
            for forbidden in (
                "import faiss", "import chromadb", "import lancedb",
                "from langchain", "import langchain", "langgraph",
                "call_llm", "run_from_config", "import subprocess",
            ):
                assert forbidden not in src, f"{mod.__name__} must not use {forbidden!r}"

    def test_indexing_calls_embeddings_only_not_chat_llm(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        # If any chat/completion path were taken, call_llm would fire. It must not.
        with patch("src.orchestration.llm.llm_interface.call_llm") as m_chat:
            api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        m_chat.assert_not_called()

    def test_indexing_does_not_execute_or_mutate_phase1(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        phase1_before = (memory_base / "memory_index.jsonl").read_bytes()
        exp_root = tmp_path / "results" / "experiments"
        exp_root.mkdir(parents=True)
        (exp_root / "exp_alpha").mkdir()
        before_dirs = {p.name for p in exp_root.iterdir()}
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        # Phase 1 index untouched; no experiment directories created.
        assert (memory_base / "memory_index.jsonl").read_bytes() == phase1_before
        assert {p.name for p in exp_root.iterdir()} == before_dirs

    def test_retrieved_items_are_compact(self, tmp_path):
        memory_base = _write_phase1(tmp_path)
        api.index_semantic_research_memory(provider="stub", memory_base=memory_base)
        result = api.semantic_retrieve_research_memory(
            query="momentum", provider="stub", memory_base=memory_base, top_k=5
        )
        allowed = {
            "memory_id", "score", "experiment_name", "artefact_type", "path",
            "context_hash", "failure_modes", "tags", "short_summary",
        }
        for it in result["items"]:
            assert set(it) <= allowed
            assert len(it["short_summary"]) <= 240
        # No raw embedding vectors leak into retrieval output.
        assert "embedding" not in json.dumps(result["items"])
