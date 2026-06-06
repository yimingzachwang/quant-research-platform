"""Tests for Phase 1 Research Memory (metadata/keyword RAG).

All tests use tmp_path for isolation.  No quant engine, no LLM, no network, no
embeddings, no vector store — Phase 1 is deterministic keyword/metadata indexing
and retrieval over compact pointers to existing artefacts.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.orchestration.api import research_api as api
from src.orchestration.memory import memory_indexer, memory_retriever, memory_store
from src.orchestration.memory.memory_schema import (
    MemoryRecord,
    compute_memory_id,
)

# ---------------------------------------------------------------------------
# Synthetic artefact tree
# ---------------------------------------------------------------------------


def _make_artefact_tree(tmp_path: Path) -> dict[str, Path]:
    """Create a small, realistic set of Zeto artefacts under tmp_path.

    Also drops a few 'must-not-index' files (parquet, secrets, plot binary) to
    prove the indexer ignores everything outside the known JSON/markdown sources.
    """
    exp_root = tmp_path / "results" / "experiments"
    llm_root = tmp_path / "results" / "llm_reviews"
    reports_root = tmp_path / "reports"
    sessions_root = tmp_path / "results" / "research_sessions"
    memory_root = tmp_path / "results" / "research_memory"

    # --- experiment A (+ a v2) -------------------------------------------
    for name, sharpe in (("exp_alpha", 0.42), ("exp_alpha_v2", 0.71)):
        d = exp_root / name
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({
            "experiment_name": name,
            "strategy_name": "MLStrategy(Ridge(alpha=0.5))",
            "parameters": {"alpha": 0.5},
            "created_at": "2026-06-01T00:00:00+00:00",
        }), encoding="utf-8")
        (d / "metrics.json").write_text(json.dumps({
            "sharpe_ratio": sharpe,
            "max_drawdown": -0.34,
            "annualized_return": 0.08,
        }), encoding="utf-8")
        # Must-not-index payloads living next to the indexable JSON.
        (d / "returns.parquet").write_text("PAR1binary", encoding="utf-8")
        (d / "secret.env").write_text("OPENAI_API_KEY=should_not_be_indexed", encoding="utf-8")
        (d / "plots").mkdir()
        (d / "plots" / "equity.png").write_text("PNGbinary", encoding="utf-8")

    # --- llm_reviews for exp_alpha (review + proposal + draft) ------------
    rdir = llm_root / "exp_alpha"
    rdir.mkdir(parents=True)
    (rdir / "llm_review.json").write_text(json.dumps({
        "experiment_name": "exp_alpha",
        "flags": ["CRITICAL: poor_oos_consistency", "CRITICAL: catastrophic_split"],
        "sections": {"recommendations": "increase regularisation"},
        "context_hash": "a" * 64,
        "generated_at": "2026-06-02T00:00:00+00:00",
    }), encoding="utf-8")
    (rdir / "iteration_proposal.json").write_text(json.dumps({
        "experiment_name": "exp_alpha",
        "research_focus": "reduce split variance via stronger regularisation",
        "rationale": "high OOS sharpe variance",
        "context_hash": "b" * 64,
        "generated_at": "2026-06-02T01:00:00+00:00",
    }), encoding="utf-8")
    (rdir / "draft_d1.json").write_text(json.dumps({
        "draft_id": "d1",
        "draft_hash": "c" * 12,
        "base_experiment": "exp_alpha",
        "proposed_name": "exp_alpha_v2",
        "changes": [{"section": "model", "field": "params.alpha",
                     "current_value": 0.5, "proposed_value": 1.0, "rationale": "x"}],
        "generated_at": "2026-06-02T02:00:00+00:00",
        "approved": False,
    }), encoding="utf-8")

    # --- report markdown --------------------------------------------------
    md_dir = reports_root / "markdown"
    md_dir.mkdir(parents=True)
    (md_dir / "exp_alpha.md").write_text(
        "# Canonical ML Showcase — validation robustness\n\nBody text...\n",
        encoding="utf-8",
    )
    # Manifest/provenance JSON must not be indexed as reports.
    (md_dir / "exp_alpha_manifest.json").write_text("{}", encoding="utf-8")

    # --- research session -------------------------------------------------
    sdir = sessions_root / "sess-1"
    sdir.mkdir(parents=True)
    (sdir / "session.json").write_text(json.dumps({
        "session_id": "sess-1",
        "research_goal": "investigate validation robustness",
        "root_experiment": "exp_alpha",
        "active_experiment": "exp_alpha_v2",
        "status": "active",
        "created_at": "2026-06-01T00:00:00+00:00",
        "updated_at": "2026-06-03T00:00:00+00:00",
        "events": [{"event_type": "review_generated"}],
        "active_draft_id": None,
    }), encoding="utf-8")

    return {
        "base": exp_root,
        "llm_base": llm_root,
        "reports_base": reports_root,
        "sessions_base": sessions_root,
        "memory_base": memory_root,
    }


def _index(tree: dict[str, Path]) -> dict:
    return api.index_research_memory(
        base=tree["base"],
        llm_base=tree["llm_base"],
        reports_base=tree["reports_base"],
        sessions_base=tree["sessions_base"],
        memory_base=tree["memory_base"],
    )


# ---------------------------------------------------------------------------
# 1. Memory item schema
# ---------------------------------------------------------------------------


class TestMemorySchema:
    def test_to_dict_has_expected_keys(self):
        rec = MemoryRecord(
            memory_id="mem_1",
            experiment_name="exp_alpha",
            artefact_type="llm_review",
            path="results/llm_reviews/exp_alpha/llm_review.json",
            short_summary="summary",
            context_hash="h" * 64,
            session_id="s1",
            failure_modes=["poor_oos_consistency"],
            tags=["validation"],
            created_at="2026-06-02T00:00:00+00:00",
        )
        d = rec.to_dict()
        assert set(d) == {
            "memory_id", "experiment_name", "session_id", "artefact_type",
            "context_hash", "path", "created_at", "failure_modes", "tags",
            "short_summary",
        }

    def test_roundtrip(self):
        rec = MemoryRecord(
            memory_id="mem_1", experiment_name="e", artefact_type="draft",
            path="p", short_summary="s", failure_modes=["fm"], tags=["t"],
        )
        assert MemoryRecord.from_dict(rec.to_dict()).to_dict() == rec.to_dict()

    def test_compute_memory_id_is_deterministic(self):
        a = compute_memory_id("llm_review", "results/llm_reviews/x/llm_review.json")
        b = compute_memory_id("llm_review", "results/llm_reviews/x/llm_review.json")
        assert a == b and a.startswith("mem_")

    def test_compute_memory_id_differs_by_type_and_path(self):
        base = compute_memory_id("llm_review", "p")
        assert compute_memory_id("draft", "p") != base
        assert compute_memory_id("llm_review", "q") != base


# ---------------------------------------------------------------------------
# 2-3. Index creation + re-index dedup
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_index_creates_records_for_each_artefact_type(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        result = _index(tree)
        records = memory_store.load_records(tree["memory_base"])
        assert result["indexed_count"] == len(records)

        types = {r.artefact_type for r in records}
        assert {
            "experiment_metadata", "experiment_metrics", "llm_review",
            "iteration_proposal", "draft", "report", "session",
        } <= types

    def test_index_writes_jsonl_index_file(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        result = _index(tree)
        index_path = Path(result["index_path"])
        assert index_path.exists()
        assert index_path.name == "memory_index.jsonl"
        # One JSON object per non-empty line.
        lines = [ln for ln in index_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == result["indexed_count"]
        for ln in lines:
            json.loads(ln)

    def test_no_duplicate_ids_on_reindex(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        first = _index(tree)
        second = _index(tree)
        records = memory_store.load_records(tree["memory_base"])
        ids = [r.memory_id for r in records]
        assert first["indexed_count"] == second["indexed_count"]
        assert len(ids) == len(set(ids))  # unique
        assert len(records) == second["indexed_count"]

    def test_does_not_index_raw_data_secrets_or_binaries(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        _index(tree)
        records = memory_store.load_records(tree["memory_base"])
        paths = [r.path for r in records]
        for p in paths:
            assert not p.endswith(".parquet")
            assert not p.endswith(".png")
            assert not p.endswith(".env")
        # No secret value leaked into any summary/tag.
        blob = json.dumps([r.to_dict() for r in records])
        assert "should_not_be_indexed" not in blob
        assert "OPENAI_API_KEY" not in blob

    def test_report_manifest_json_not_indexed_as_report(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        _index(tree)
        records = memory_store.load_records(tree["memory_base"])
        report_paths = [r.path for r in records if r.artefact_type == "report"]
        assert any(p.endswith("exp_alpha.md") for p in report_paths)
        assert all(p.endswith(".md") for p in report_paths)

    def test_failure_modes_extracted_from_review_flags(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        _index(tree)
        records = memory_store.load_records(tree["memory_base"])
        review = next(r for r in records if r.artefact_type == "llm_review")
        assert "poor_oos_consistency" in review.failure_modes
        assert "catastrophic_split" in review.failure_modes
        # Severity prefix is stripped.
        assert all(":" not in fm for fm in review.failure_modes)

    def test_indexing_does_not_mutate_or_execute(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        before = {
            p: p.read_bytes()
            for p in tree["base"].rglob("*") if p.is_file()
        }
        exp_dirs_before = {p.name for p in tree["base"].iterdir()}
        _index(tree)
        # Source artefacts unchanged.
        after = {p: p.read_bytes() for p in tree["base"].rglob("*") if p.is_file()}
        assert before == after
        # No new experiment directories created (no execution happened).
        assert {p.name for p in tree["base"].iterdir()} == exp_dirs_before


# ---------------------------------------------------------------------------
# 4-8. Retrieval
# ---------------------------------------------------------------------------


def _records(tmp_path: Path) -> list[MemoryRecord]:
    tree = _make_artefact_tree(tmp_path)
    _index(tree)
    return memory_store.load_records(tree["memory_base"])


class TestRetrieval:
    def test_retrieve_by_experiment_name_includes_versions(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(recs, experiment_name="exp_alpha", top_k=50)
        assert items
        names = {it["experiment_name"] for it in items}
        # Base name matches itself and its versioned descendant.
        assert "exp_alpha" in names
        assert "exp_alpha_v2" in names

    def test_retrieve_by_failure_mode(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(
            recs, failure_modes=["poor_oos_consistency", "catastrophic_split"], top_k=10
        )
        assert items
        for it in items:
            assert set(it["failure_modes"]) & {"poor_oos_consistency", "catastrophic_split"}
            assert it["matched_terms"]

    def test_retrieve_by_artefact_type(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(recs, artefact_type="iteration_proposal", top_k=10)
        assert items
        assert all(it["artefact_type"] == "iteration_proposal" for it in items)

    def test_retrieve_by_artefact_type_is_case_insensitive(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(recs, artefact_type="LLM_REVIEW", top_k=10)
        assert items and all(it["artefact_type"] == "llm_review" for it in items)

    def test_retrieve_by_keyword(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(recs, query="regularisation", top_k=10)
        assert items
        for it in items:
            assert it["matched_terms"]

    def test_retrieve_keyword_no_match_returns_empty(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(recs, query="zzz_no_such_token", top_k=10)
        assert items == []

    def test_top_k_limits_results(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(recs, experiment_name="exp_alpha", top_k=2)
        assert len(items) <= 2

    def test_empty_index_behaviour(self, tmp_path):
        # No index file at all.
        memory_base = tmp_path / "results" / "research_memory"
        status = api.get_research_memory_status(memory_base=memory_base)
        assert status["index_exists"] is False
        assert status["item_count"] == 0
        assert status["experiment_count"] == 0
        items = api.retrieve_research_memory(
            query="anything", failure_modes=["poor_oos_consistency"], memory_base=memory_base
        )
        assert items == []

    def test_retrieved_items_are_compact(self, tmp_path):
        recs = _records(tmp_path)
        items = memory_retriever.retrieve_memory(recs, experiment_name="exp_alpha", top_k=5)
        allowed = {
            "memory_id", "experiment_name", "artefact_type", "path",
            "context_hash", "failure_modes", "tags", "matched_terms",
            "short_summary",
        }
        for it in items:
            assert set(it) <= allowed
            # Compact: short summary bounded; no full artefact body.
            assert len(it["short_summary"]) <= 240


# ---------------------------------------------------------------------------
# Research API status / index counts
# ---------------------------------------------------------------------------


class TestResearchApiMemory:
    def test_status_after_index(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        result = _index(tree)
        status = api.get_research_memory_status(memory_base=tree["memory_base"])
        assert status["index_exists"] is True
        assert status["item_count"] == result["indexed_count"]
        assert status["experiment_count"] >= 2

    def test_retrieve_via_research_api(self, tmp_path):
        tree = _make_artefact_tree(tmp_path)
        _index(tree)
        items = api.retrieve_research_memory(
            failure_modes=["poor_oos_consistency"],
            memory_base=tree["memory_base"],
            top_k=5,
        )
        assert items
        assert all("poor_oos_consistency" in it["failure_modes"] for it in items)


# ---------------------------------------------------------------------------
# Phase-1 constraints: no embeddings / vector store / LLM / execution
# ---------------------------------------------------------------------------


class TestPhase1Constraints:
    def test_no_embedding_or_vector_or_llm_imports(self):
        # Target real import/usage tokens, not the prose that documents the
        # Phase-1 non-goals ("no embeddings, no vector store, ...").
        for mod in (memory_indexer, memory_retriever, memory_store):
            src = inspect.getsource(mod)
            for forbidden in (
                "import faiss", "import chromadb", "import lancedb",
                "from langchain", "import langchain", "langgraph",
                "sentence_transformers", "OpenAIEmbeddings",
                ".embed(", "embed_query", "embed_documents",
                "call_llm", "import subprocess", "run_from_config",
            ):
                assert forbidden not in src, f"{mod.__name__} must not use {forbidden!r}"

    def test_indexer_does_not_import_quant_engine(self):
        src = inspect.getsource(memory_indexer)
        for forbidden in ("from src.experiments", "from src.backtesting",
                          "from src.models", "from src.data"):
            assert forbidden not in src
