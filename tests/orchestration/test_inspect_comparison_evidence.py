"""Tests for inspect_comparison_evidence API + MCP tool.

Covers all 8 required scenarios:
  1. Successful inspection of a synthetic evidence record
  2. Missing evidence returns ok=false, stage="comparison_evidence_not_found"
  3. Tool signature accepts only base/candidate names (no arbitrary path arg)
  4. No LLM call during inspection
  5. No approval/render/execute side effects
  6. Compact MCP envelope: contract keys present, stage correct, evidence_path in data
  7. Operator manual contains routing rule for inspect_comparison_evidence
  8. Existing test_zeto_mcp_server tests still pass (verified by running full suite)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

from src.mcp import zeto_server as zeto
from src.orchestration.api import research_api as api
from src.orchestration.utils.filesystem import comparison_evidence_json_path

_MCP_MOD = "src.mcp.zeto_server._api"
_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}

_SYNTHETIC_EVIDENCE = {
    "base_experiment_name": "base_exp",
    "candidate_experiment_name": "cand_exp",
    "research_question": "Does risk-adjusted momentum improve OOS stability?",
    "tested_change": "added risk_adjusted_momentum_20",
    "base_metrics": {"sharpe_ratio": 0.695, "max_drawdown": -0.341},
    "candidate_metrics": {"sharpe_ratio": 0.704, "max_drawdown": -0.341},
    "metric_deltas": {
        "delta_sharpe": 0.009,
        "delta_mean_oos_sharpe": -0.207,
        "delta_max_drawdown_pct": 0.0,
    },
    "failure_modes_base": ["poor_oos_consistency"],
    "failure_modes_candidate": ["poor_oos_consistency"],
    "conclusion": "Candidate shows marginal Sharpe gain but worse OOS stability.",
    "session_id": "sess-abc",
    "created_at": "2026-06-06T00:00:00+00:00",
}


def _write_evidence(tmp_path: Path, evidence: dict | None = None) -> Path:
    """Write synthetic evidence JSON; return its path."""
    path = comparison_evidence_json_path("base_exp", "cand_exp", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence or _SYNTHETIC_EVIDENCE))
    return path


# ---------------------------------------------------------------------------
# 1. Successful inspection
# ---------------------------------------------------------------------------


def test_successful_inspection_returns_ok(tmp_path):
    path = _write_evidence(tmp_path)
    result = api.inspect_comparison_evidence(
        "base_exp", "cand_exp", comparisons_base=tmp_path
    )
    assert result["status"] == "ok"
    assert result["base_experiment_name"] == "base_exp"
    assert result["candidate_experiment_name"] == "cand_exp"
    assert result["tested_change"] == "added risk_adjusted_momentum_20"
    assert result["research_question"] == "Does risk-adjusted momentum improve OOS stability?"
    assert isinstance(result["metric_deltas"], dict)
    assert result["metric_deltas"]["delta_sharpe"] == pytest.approx(0.009)
    assert result["failure_modes_base"] == ["poor_oos_consistency"]
    assert result["failure_modes_candidate"] == ["poor_oos_consistency"]
    assert "evidence_path" in result
    assert "base_exp__vs__cand_exp" in result["evidence_path"]


def test_successful_inspection_returns_conclusion(tmp_path):
    _write_evidence(tmp_path)
    result = api.inspect_comparison_evidence(
        "base_exp", "cand_exp", comparisons_base=tmp_path
    )
    assert result["conclusion"] != ""


# ---------------------------------------------------------------------------
# 2. Missing evidence → ok=false, stage="comparison_evidence_not_found"
# ---------------------------------------------------------------------------


def test_missing_evidence_api_returns_not_found(tmp_path):
    result = api.inspect_comparison_evidence(
        "base_exp", "cand_exp", comparisons_base=tmp_path
    )
    assert result["status"] == "not_found"
    assert len(result["errors"]) > 0
    assert "compare_experiment_metrics" in result["errors"][0]


def test_missing_evidence_mcp_returns_not_found_envelope():
    stub = {
        "status": "not_found",
        "base_experiment_name": "base_exp",
        "candidate_experiment_name": "cand_exp",
        "evidence_path": "results/comparisons/base_exp__vs__cand_exp/comparison_evidence.json",
        "errors": ["No comparison evidence found for 'base_exp' vs 'cand_exp'."],
    }
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub):
        out = zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    assert out["ok"] is False
    assert out["stage"] == "comparison_evidence_not_found"
    assert out["next_suggested_action"] == "compare_experiment_metrics"
    assert "base_exp" in out["display"]
    assert "cand_exp" in out["display"]


# ---------------------------------------------------------------------------
# 3. No arbitrary path argument in the MCP tool signature
# ---------------------------------------------------------------------------


def test_mcp_tool_accepts_only_name_args():
    sig = inspect.signature(zeto.inspect_comparison_evidence)
    param_names = set(sig.parameters)
    assert param_names == {"base_experiment_name", "candidate_experiment_name"}, (
        f"MCP tool must not expose arbitrary path params; got {param_names}"
    )


# ---------------------------------------------------------------------------
# 4. No LLM call
# ---------------------------------------------------------------------------


def test_no_llm_call_on_successful_inspection(tmp_path):
    _write_evidence(tmp_path)
    stub_ok = dict(_SYNTHETIC_EVIDENCE, status="ok", evidence_path="x/comparison_evidence.json")
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub_ok) as m_api, \
         patch("src.orchestration.config_generation.draft_generator.call_llm") as m_llm:
        zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    m_llm.assert_not_called()


def test_no_llm_call_on_missing_evidence():
    stub_nf = {
        "status": "not_found",
        "base_experiment_name": "base_exp",
        "candidate_experiment_name": "cand_exp",
        "evidence_path": "x/comparison_evidence.json",
        "errors": ["No evidence found."],
    }
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub_nf), \
         patch("src.orchestration.config_generation.draft_generator.call_llm") as m_llm:
        zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    m_llm.assert_not_called()


# ---------------------------------------------------------------------------
# 5. No approval/render/execute side effects
# ---------------------------------------------------------------------------


def test_no_approval_render_execute_side_effects(tmp_path):
    stub_ok = dict(_SYNTHETIC_EVIDENCE, status="ok", evidence_path="x/comparison_evidence.json")
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub_ok), \
         patch(f"{_MCP_MOD}.approve_experiment_draft") as m_approve, \
         patch(f"{_MCP_MOD}.render_draft_to_yaml") as m_render, \
         patch(f"{_MCP_MOD}.execute_approved_config") as m_exec:
        zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    m_approve.assert_not_called()
    m_render.assert_not_called()
    m_exec.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Compact MCP envelope contract
# ---------------------------------------------------------------------------


def test_successful_envelope_contract():
    stub_ok = dict(_SYNTHETIC_EVIDENCE, status="ok", evidence_path="x/comparison_evidence.json")
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub_ok):
        out = zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    assert _CONTRACT_KEYS.issubset(out), f"Missing keys: {_CONTRACT_KEYS - set(out)}"
    assert out["ok"] is True
    assert out["stage"] == "comparison_evidence_inspected"
    assert out["next_suggested_action"] is None


def test_successful_envelope_contains_evidence_path_in_data():
    stub_ok = dict(_SYNTHETIC_EVIDENCE, status="ok", evidence_path="results/comparisons/x/comparison_evidence.json")
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub_ok):
        out = zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    assert "evidence_path" in out["data"]


def test_successful_envelope_display_mentions_experiments():
    stub_ok = dict(_SYNTHETIC_EVIDENCE, status="ok", evidence_path="x/comparison_evidence.json")
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub_ok):
        out = zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    assert "base_exp" in out["display"]
    assert "cand_exp" in out["display"]


def test_not_found_envelope_contract():
    stub_nf = {
        "status": "not_found",
        "base_experiment_name": "base_exp",
        "candidate_experiment_name": "cand_exp",
        "evidence_path": "x/comparison_evidence.json",
        "errors": ["No evidence found."],
    }
    with patch(f"{_MCP_MOD}.inspect_comparison_evidence", return_value=stub_nf):
        out = zeto.inspect_comparison_evidence("base_exp", "cand_exp")

    assert _CONTRACT_KEYS.issubset(out)
    assert out["ok"] is False
    assert out["stage"] == "comparison_evidence_not_found"
    assert out["next_suggested_action"] == "compare_experiment_metrics"


# ---------------------------------------------------------------------------
# 7. Operator manual routing rule
# ---------------------------------------------------------------------------


def test_operator_manual_contains_inspect_comparison_evidence_routing():
    combined = " ".join(zeto._OPERATOR_RULES)
    assert "inspect_comparison_evidence" in combined, (
        "Operator manual must contain routing rule for inspect_comparison_evidence"
    )


def test_operator_manual_routing_covers_retrieve_memory_fallback():
    combined = " ".join(zeto._OPERATOR_RULES)
    assert "retrieve_research_memory" in combined
    assert "artefact_type" in combined or "comparison_evidence" in combined


import pytest
