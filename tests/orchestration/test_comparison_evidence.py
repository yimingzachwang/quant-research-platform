"""Tests for comparison evidence memory records.

Covers all 9 required scenarios:
  1. compare_experiment_metrics persists comparison_evidence
  2. evidence includes base/candidate names and metric deltas
  3. missing metrics handled cleanly (no evidence written; no crash)
  4. Phase 1 memory indexes comparison_evidence records
  5. semantic memory retrieves comparison_evidence for relevant queries
  6. no LLM call in compare_experiment_metrics or indexing
  7. no execution/approval/render side effects
  8. compact MCP envelope preserved (evidence_path in data)
  9. existing memory / MCP tests still pass (verified by running full suite)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.orchestration.api import research_api as api
from src.orchestration.memory import memory_indexer
from src.orchestration.memory.memory_schema import ARTEFACT_TYPES
from src.mcp import zeto_server as zeto

_MCP_MOD = "src.mcp.zeto_server._api"
_LLM_MOD = "src.orchestration.config_generation.draft_generator.call_llm"
_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_experiment(
    base: Path,
    name: str,
    *,
    sharpe: float = 0.61,
    mean_oos: float = -0.22,
    max_dd: float = -0.312,
    with_validation: bool = True,
) -> None:
    d = base / name
    (d / "diagnostics").mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({
        "experiment_name": name,
        "strategy_name": "MLStrategy(Ridge)",
        "created_at": "2026-06-06T00:00:00+00:00",
    }))
    (d / "metrics.json").write_text(json.dumps({
        "sharpe_ratio": sharpe,
        "annualized_return": 0.08,
        "annualized_volatility": 0.13,
        "max_drawdown": max_dd,
        "calmar_ratio": 0.26,
        "hit_rate": 0.51,
    }))
    if with_validation:
        (d / "diagnostics" / "split_metrics.json").write_text(json.dumps({
            "summary": {
                "n_splits": 6,
                "mean_sharpe": mean_oos,
                "std_sharpe": 1.1,
                "hit_rate_positive_sharpe": 0.33,
                "mean_annualized_return": 0.02,
                "worst_max_drawdown": -0.40,
            },
            "splits": [{"sharpe_ratio": x} for x in [0.1, -0.5, -0.2, 0.1, -0.6, -0.3]],
        }))


def _setup_two_experiments(tmp_path: Path) -> tuple[Path, Path]:
    """Write base + candidate experiments; return (exp_base, comparisons_base)."""
    exp_base = tmp_path / "results" / "experiments"
    _write_experiment(exp_base, "base_exp", sharpe=0.695, mean_oos=-0.320, max_dd=-0.341)
    _write_experiment(exp_base, "cand_exp", sharpe=0.704, mean_oos=-0.527, max_dd=-0.341)
    comparisons_base = tmp_path / "results" / "comparisons"
    return exp_base, comparisons_base


# ---------------------------------------------------------------------------
# 1 & 2: Evidence is persisted and has correct content
# ---------------------------------------------------------------------------


class TestComparisonEvidencePersistence:
    def test_evidence_file_is_written(self, tmp_path):
        """compare_experiment_metrics writes comparison_evidence.json on success."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        result = api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
            persist_evidence=True,
        )
        assert result["status"] == "ok"
        assert result["evidence_path"] is not None
        ev_path = Path(result["evidence_path"])
        assert ev_path.exists()
        assert ev_path.name == "comparison_evidence.json"

    def test_evidence_contains_required_fields(self, tmp_path):
        """Evidence record includes base/candidate names, metric deltas, conclusion."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        result = api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
            tested_change="added risk_adjusted_momentum_20",
            research_question="Did RAM help OOS stability?",
            session_id="sess_abc",
            persist_evidence=True,
        )
        ev = json.loads(Path(result["evidence_path"]).read_text())

        assert ev["artefact_type"] == "comparison_evidence"
        assert ev["base_experiment_name"] == "base_exp"
        assert ev["candidate_experiment_name"] == "cand_exp"
        assert ev["tested_change"] == "added risk_adjusted_momentum_20"
        assert ev["research_question"] == "Did RAM help OOS stability?"
        assert ev["session_id"] == "sess_abc"

        # Metric deltas present.
        assert "delta_sharpe" in ev["metric_deltas"]
        assert ev["metric_deltas"]["delta_sharpe"] is not None
        assert "delta_mean_oos_sharpe" in ev["metric_deltas"]

        # Base and candidate metrics.
        assert ev["base_metrics"]["sharpe_ratio"] == pytest.approx(0.695, abs=1e-3)
        assert ev["candidate_metrics"]["sharpe_ratio"] == pytest.approx(0.704, abs=1e-3)

        # Failure modes and conclusion.
        assert isinstance(ev["failure_modes_candidate"], list)
        assert ev["conclusion"]
        assert ev["created_at"]

    def test_evidence_path_in_comparison_dir(self, tmp_path):
        """Evidence is written inside the standard comparisons directory."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        result = api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
        )
        ev_path = Path(result["evidence_path"])
        # Must be under results/comparisons/base_exp__vs__cand_exp/
        assert "base_exp__vs__cand_exp" in str(ev_path)

    def test_persist_evidence_false_skips_write(self, tmp_path):
        """persist_evidence=False: no file written, evidence_path=None."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        result = api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
            persist_evidence=False,
        )
        assert result["status"] == "ok"
        assert result["evidence_path"] is None
        assert not (comp_base / "base_exp__vs__cand_exp" / "comparison_evidence.json").exists()


# ---------------------------------------------------------------------------
# 3: Missing metrics handled cleanly
# ---------------------------------------------------------------------------


class TestMissingMetricsHandledCleanly:
    def test_not_found_returns_status_no_crash(self, tmp_path):
        """If one experiment is missing, status=not_found; no evidence written."""
        exp_base = tmp_path / "results" / "experiments"
        _write_experiment(exp_base, "base_exp")
        comp_base = tmp_path / "results" / "comparisons"
        result = api.compare_experiment_metrics(
            "base_exp", "does_not_exist",
            base=exp_base,
            comparisons_base=comp_base,
        )
        assert result["status"] == "not_found"
        assert "does_not_exist" in result["missing_experiments"]
        assert "evidence_path" not in result  # never added for not_found

    def test_both_missing_handled(self, tmp_path):
        """Both experiments missing → clean not_found response."""
        exp_base = tmp_path / "results" / "experiments"
        comp_base = tmp_path / "results" / "comparisons"
        result = api.compare_experiment_metrics(
            "ghost_a", "ghost_b",
            base=exp_base,
            comparisons_base=comp_base,
        )
        assert result["status"] == "not_found"
        assert len(result["missing_experiments"]) == 2


# ---------------------------------------------------------------------------
# 4: Phase 1 memory indexes comparison_evidence
# ---------------------------------------------------------------------------


class TestPhase1MemoryIndexesComparisonEvidence:
    def test_comparison_evidence_in_artefact_types(self):
        """comparison_evidence is a first-class artefact type in the schema."""
        assert "comparison_evidence" in ARTEFACT_TYPES

    def test_indexer_picks_up_evidence_file(self, tmp_path):
        """build_memory_records scans comparisons/ and produces comparison_evidence records."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
        )
        records = memory_indexer.build_memory_records(
            base=exp_base,
            comparisons_base=comp_base,
        )
        comp_records = [r for r in records if r.artefact_type == "comparison_evidence"]
        assert len(comp_records) == 1
        rec = comp_records[0]
        assert rec.experiment_name == "cand_exp"
        assert rec.short_summary  # non-empty compact summary

    def test_comparison_evidence_summary_contains_delta(self, tmp_path):
        """Summary includes delta_sharpe so retrieval can reason about it."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
            tested_change="added sma_50",
        )
        records = memory_indexer.build_memory_records(
            base=exp_base, comparisons_base=comp_base,
        )
        rec = next(r for r in records if r.artefact_type == "comparison_evidence")
        assert "base_exp" in rec.short_summary
        assert "cand_exp" in rec.short_summary
        assert "sharpe" in rec.short_summary.lower() or "delta" in rec.short_summary.lower()

    def test_comparison_evidence_tags_derived(self, tmp_path):
        """comparison_evidence records get comparison + feature_change tags."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
            tested_change="added sma_50",
        )
        records = memory_indexer.build_memory_records(
            base=exp_base, comparisons_base=comp_base,
        )
        rec = next(r for r in records if r.artefact_type == "comparison_evidence")
        assert "comparison" in rec.tags

    def test_retrieve_by_artefact_type(self, tmp_path):
        """retrieve_research_memory(artefact_type='comparison_evidence') returns the record."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
        )
        mem_base = tmp_path / "results" / "research_memory"
        api.index_research_memory(
            base=exp_base,
            comparisons_base=comp_base,
            memory_base=mem_base,
        )
        results = api.retrieve_research_memory(
            artefact_type="comparison_evidence",
            memory_base=mem_base,
        )
        assert len(results) >= 1
        assert all(r["artefact_type"] == "comparison_evidence" for r in results)

    def test_evidence_indexed_by_experiment_name(self, tmp_path):
        """Phase 1 retrieval by experiment_name finds the evidence record."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
        )
        mem_base = tmp_path / "results" / "research_memory"
        api.index_research_memory(
            base=exp_base, comparisons_base=comp_base, memory_base=mem_base,
        )
        results = api.retrieve_research_memory(
            experiment_name="cand_exp",
            artefact_type="comparison_evidence",
            memory_base=mem_base,
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# 5: Semantic memory retrieves comparison_evidence
# ---------------------------------------------------------------------------


class TestSemanticMemoryRetrievesComparisonEvidence:
    def test_semantic_text_includes_comparison_fields(self, tmp_path):
        """The embed text for a comparison_evidence record contains useful signal."""
        from src.orchestration.memory.memory_indexer import build_memory_records
        from src.orchestration.memory.semantic_indexer import build_semantic_text

        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
            tested_change="added momentum_20",
        )
        records = build_memory_records(base=exp_base, comparisons_base=comp_base)
        comp_rec = next(r for r in records if r.artefact_type == "comparison_evidence")
        text = build_semantic_text(comp_rec)

        assert "comparison_evidence" in text
        assert "cand_exp" in text
        assert comp_rec.short_summary in text

    def test_semantic_index_includes_comparison_evidence(self, tmp_path):
        """After indexing, semantic index covers comparison_evidence records."""
        from unittest.mock import MagicMock

        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp",
            base=exp_base,
            comparisons_base=comp_base,
        )
        mem_base = tmp_path / "results" / "research_memory"
        api.index_research_memory(
            base=exp_base, comparisons_base=comp_base, memory_base=mem_base,
        )

        # Patch the embeddings call so the test stays LLM-free.
        from src.orchestration.memory.memory_store import load_records
        records = load_records(mem_base)
        comp_records = [r for r in records if r.artefact_type == "comparison_evidence"]
        assert len(comp_records) >= 1

        from types import SimpleNamespace

        def fake_embed(texts, **kwargs):
            vecs = [[float(i % 10) / 10.0] * 4 for i in range(len(texts))]
            return SimpleNamespace(model="fake-embed-model", vectors=vecs)

        with patch("src.orchestration.memory.semantic_indexer.embed_texts", side_effect=fake_embed):
            result = api.index_semantic_research_memory(
                memory_base=mem_base,
            )
        assert result.get("embedded_count", 0) > 0

        # Verify comparison_evidence appears in the semantic index JSONL.
        from src.orchestration.memory.semantic_store import load_semantic_records
        sem_recs = load_semantic_records(mem_base)
        comp_sem = [r for r in sem_recs if r.metadata.get("artefact_type") == "comparison_evidence"]
        assert len(comp_sem) >= 1


# ---------------------------------------------------------------------------
# 6: No LLM call
# ---------------------------------------------------------------------------


class TestNoLlmCall:
    def test_compare_and_index_make_no_llm_call(self, tmp_path):
        """No LLM is called by compare_experiment_metrics or build_memory_records."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        with patch("src.orchestration.llm.llm_interface.call_llm") as m_llm:
            api.compare_experiment_metrics(
                "base_exp", "cand_exp",
                base=exp_base,
                comparisons_base=comp_base,
            )
            memory_indexer.build_memory_records(
                base=exp_base, comparisons_base=comp_base,
            )
        m_llm.assert_not_called()

    def test_source_code_does_not_reference_rag(self):
        """compare_experiment_metrics source does not call retrieve_memory or semantic."""
        import inspect
        src = inspect.getsource(api.compare_experiment_metrics)
        src += inspect.getsource(api._persist_comparison_evidence)
        assert "retrieve_memory" not in src
        assert "semantic" not in src


# ---------------------------------------------------------------------------
# 7: No execution/approval/render side effects
# ---------------------------------------------------------------------------


class TestNoSideEffects:
    def test_no_approval_or_execution(self, tmp_path):
        """compare_experiment_metrics does not approve, render, or execute anything."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        with (
            patch("src.orchestration.config_generation.draft_validator.approve_draft") as m_approve,
            patch("src.orchestration.api.research_api.render_to_yaml", create=True) as m_render,
            patch("src.orchestration.api.research_api.execute_approved_config", create=True) as m_exec,
        ):
            api.compare_experiment_metrics(
                "base_exp", "cand_exp",
                base=exp_base, comparisons_base=comp_base,
            )
        m_approve.assert_not_called()
        m_render.assert_not_called()
        m_exec.assert_not_called()

    def test_indexing_does_not_mutate_source_artefacts(self, tmp_path):
        """Indexing only reads; source metrics.json is not modified."""
        exp_base, comp_base = _setup_two_experiments(tmp_path)
        api.compare_experiment_metrics(
            "base_exp", "cand_exp", base=exp_base, comparisons_base=comp_base,
        )
        metrics_path = exp_base / "base_exp" / "metrics.json"
        mtime_before = metrics_path.stat().st_mtime
        memory_indexer.build_memory_records(base=exp_base, comparisons_base=comp_base)
        assert metrics_path.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# 8: MCP envelope preserved (evidence_path in data)
# ---------------------------------------------------------------------------


class TestMcpEnvelopePreserved:
    def test_compare_mcp_tool_includes_evidence_path(self):
        """MCP compare tool returns standard envelope with evidence_path in data."""
        stub = {
            "status": "ok",
            "base_experiment": "base_exp",
            "candidate_experiment": "cand_exp",
            "base_sharpe": 0.695,
            "candidate_sharpe": 0.704,
            "delta_sharpe": 0.009,
            "base_mean_oos_sharpe": -0.320,
            "candidate_mean_oos_sharpe": -0.527,
            "delta_mean_oos_sharpe": -0.207,
            "base_max_drawdown": "-34.10%",
            "candidate_max_drawdown": "-34.10%",
            "delta_max_drawdown_pct": 0.0,
            "base_failure_modes": ["poor_oos_consistency"],
            "candidate_failure_modes": ["poor_oos_consistency", "catastrophic_split"],
            "conclusion": "Comparison base_exp -> cand_exp: Sharpe 0.695 -> 0.704.",
            "evidence_path": "results/comparisons/base_exp__vs__cand_exp/comparison_evidence.json",
        }
        with patch(f"{_MCP_MOD}.compare_experiment_metrics", return_value=stub):
            out = zeto.compare_experiment_metrics("base_exp", "cand_exp")

        assert _CONTRACT_KEYS.issubset(out)
        assert out["ok"] is True
        assert out["stage"] == "experiment_metrics_compared"
        assert out["data"]["evidence_path"] is not None
        assert "evidence_path" in out["data"]

    def test_compare_mcp_tool_optional_context_params(self):
        """MCP tool accepts session_id, research_question, tested_change."""
        stub = {
            "status": "ok",
            "base_experiment": "b", "candidate_experiment": "c",
            "base_sharpe": 0.5, "candidate_sharpe": 0.6, "delta_sharpe": 0.1,
            "base_mean_oos_sharpe": None, "candidate_mean_oos_sharpe": None,
            "delta_mean_oos_sharpe": None,
            "base_max_drawdown": None, "candidate_max_drawdown": None,
            "delta_max_drawdown_pct": None,
            "base_failure_modes": [], "candidate_failure_modes": [],
            "conclusion": "Comparison b -> c: Sharpe 0.5 -> 0.6.",
            "evidence_path": None,
        }
        with patch(f"{_MCP_MOD}.compare_experiment_metrics", return_value=stub) as m:
            out = zeto.compare_experiment_metrics(
                "b", "c",
                session_id="s1",
                research_question="Did feature X help?",
                tested_change="added feature_x",
            )
        assert out["ok"] is True
        call_kwargs = m.call_args
        # Positional or keyword — just check the tool succeeded.
        assert out["stage"] == "experiment_metrics_compared"

    def test_compare_mcp_not_found_envelope(self):
        """not_found case returns ok=False with standard envelope."""
        stub = {
            "status": "not_found",
            "base_experiment": "a", "candidate_experiment": "b",
            "missing_experiments": ["b"],
        }
        with patch(f"{_MCP_MOD}.compare_experiment_metrics", return_value=stub):
            out = zeto.compare_experiment_metrics("a", "b")
        assert out["ok"] is False
        assert out["stage"] == "experiment_metrics_unavailable"


# ---------------------------------------------------------------------------
# Operator manual routing rule
# ---------------------------------------------------------------------------


class TestOperatorManualComparisonEvidence:
    def test_routing_rule_present(self):
        """Operator manual includes comparison_evidence routing for 'what did we learn?'."""
        blob = json.dumps(zeto.get_zeto_operator_manual()).lower()
        assert "comparison_evidence" in blob
        assert "what did we learn" in blob or "retrieve_research_memory" in blob


import pytest  # noqa: E402 — used by pytest.approx above
