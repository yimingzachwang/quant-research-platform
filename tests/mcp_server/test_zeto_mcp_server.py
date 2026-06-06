"""Tests for the Zeto MCP server.

All Research API calls are mocked — no disk I/O, no quant engine, no LLM, no
real experiments, and LM Studio is never required.  These tests verify:

  * the visible-state contract (ok/stage/display/data/next_suggested_action)
    on every tool response, with the required content in key displays
  * tool wrappers delegate to the Research API (not engine internals)
  * the execution confirmation gate and no implicit approve/render
  * that no all-in-one / silent full-loop tool exists
  * non-coupling (no shell/subprocess/quant-engine internals)
"""

from __future__ import annotations

import inspect
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from src.mcp import zeto_server as zeto
from src.orchestration.api.research_api import ExecutionResult
from src.orchestration.config_generation.draft_schema import (
    DraftChange,
    DraftValidationResult,
    ExperimentDraft,
)
from src.orchestration.session.session_schema import ResearchSession, SessionStatus

_MOD = "src.mcp.zeto_server._api"

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}

_EXPECTED_TOOLS = {
    "get_zeto_operator_manual",
    "list_experiments",
    "create_research_session",
    "get_session_summary",
    "list_research_sessions",
    "get_latest_research_session",
    "check_research_workflow_state",
    "build_context_summary",
    "run_experiment_review",
    "generate_iteration_proposal",
    "generate_experiment_draft",
    "validate_experiment_draft",
    "approve_experiment_draft",
    "render_draft_to_yaml",
    "execute_approved_config",
    "review_post_run_result",
}


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _stub_session() -> ResearchSession:
    return ResearchSession(
        session_id="s1",
        research_goal="goal",
        root_experiment="exp_a",
        active_experiment="exp_a",
        status=SessionStatus.ACTIVE,
        created_at="2026-05-30T00:00:00+00:00",
        updated_at="2026-05-30T00:00:00+00:00",
    )


def _stub_draft(approved: bool = False) -> ExperimentDraft:
    return ExperimentDraft(
        draft_id="d1",
        draft_hash="hash",
        base_experiment="exp_a",
        source_proposal_hash="p1",
        proposed_name="exp_a_v2",
        changes=[
            DraftChange(
                section="model",
                field="params.alpha",
                current_value=0.5,
                proposed_value=1.0,
                rationale="stronger regularisation",
            )
        ],
        generated_at="2026-05-30T00:00:00+00:00",
        approved=approved,
        approved_at="2026-05-30T00:01:00+00:00" if approved else None,
    )


def _stub_summary() -> dict:
    return {
        "session_id": "s1",
        "event_count": 1,
        "active_experiment": "exp_a",
        "active_draft": None,
        "approved_config_path": None,
        "research_goal": "goal",
        "experiments_visited": ["exp_a"],
        "status": "active",
    }


def _stub_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        experiment_name="exp_a",
        failure_modes=[
            {"name": "poor_oos_consistency", "severity": "critical", "description": "neg OOS"},
        ],
        performance={"sharpe": 0.42},
        validation={"mean_oos_sharpe": -0.1, "std_oos_sharpe": 0.8, "n_splits": 6},
    )


def _stub_review() -> SimpleNamespace:
    return SimpleNamespace(
        sections={"performance_interpretation": "x", "recommendations": "y"},
        flags=["CRITICAL: poor_oos_consistency"],
    )


def _stub_proposal() -> SimpleNamespace:
    return SimpleNamespace(
        research_focus="reduce split variance",
        rationale="high std OOS sharpe",
        supporting_evidence=["std_oos_sharpe=0.8"],
        suggested_experiments=["longer train window"],
        validation_concerns=["catastrophic split"],
        confidence="medium",
        context_hash="c" * 64,
    )


def _stub_execution(success: bool = True) -> ExecutionResult:
    return ExecutionResult(
        config_path="configs/experiments/exp_a_v2.yaml",
        experiment_name="exp_a_v2" if success else None,
        success=success,
        artefact_root="results/experiments/exp_a_v2" if success else None,
        report_path=None,
        error=None if success else "boom",
        command_hint="python scripts/run_from_config.py configs/experiments/exp_a_v2.yaml",
    )


def _stub_state(**overrides) -> dict:
    state = {
        "experiment_name": "exp_a",
        "baseline_artefacts_exist": True,
        "metadata_exists": True,
        "metrics_exists": True,
        "context_ready": True,
        "review_exists": False,
        "proposal_exists": False,
        "draft_exists": False,
        "latest_draft_id": None,
        "latest_draft_approved": False,
        "proposed_name": None,
        "rendered_yaml_exists": False,
        "rendered_yaml_path": None,
        "revised_artefacts_exist": False,
        "report_path": None,
        "plots_dir": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Server / tool registration
# ---------------------------------------------------------------------------


def test_server_imports_and_has_name():
    assert zeto.mcp is not None
    assert zeto.mcp.name == "zeto"


def test_tool_list_contains_expected_names():
    names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
    assert names == _EXPECTED_TOOLS


def test_every_tool_has_a_description():
    for t in zeto.mcp._tool_manager.list_tools():
        assert t.description and t.description.strip()


def test_no_all_in_one_or_loop_tool_exists():
    names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
    # Granular tools only — nothing that runs the whole flow silently.
    # (check_research_workflow_state is a read-only preflight, not a run loop.)
    assert len(names) == 16
    for forbidden in ("auto", "loop", "full", "pipeline", "run_all", "orchestrate", "everything"):
        assert not any(forbidden in n.lower() for n in names), f"loop-like tool name: {forbidden}"


# ---------------------------------------------------------------------------
# Operator manual (fixed, read-only)
# ---------------------------------------------------------------------------


def test_operator_manual_is_registered():
    names = {t.name for t in zeto.mcp._tool_manager.list_tools()}
    assert "get_zeto_operator_manual" in names


def test_operator_manual_returns_envelope():
    out = zeto.get_zeto_operator_manual()
    assert _CONTRACT_KEYS.issubset(out)
    assert out["ok"] is True
    assert out["stage"] == "operator_manual_loaded"
    assert out["next_suggested_action"] == "check_research_workflow_state"
    assert out["data"]["manual_path"] == "docs/LM_STUDIO_QWEN_OPERATOR_MANUAL.md"
    assert isinstance(out["data"]["rules"], list) and out["data"]["rules"]


def test_operator_manual_is_compact():
    assert len(json.dumps(zeto.get_zeto_operator_manual())) < 4000


def test_operator_manual_includes_key_rules():
    blob = json.dumps(zeto.get_zeto_operator_manual()).lower()
    assert "snake_case" in blob
    assert "check_research_workflow_state" in blob
    assert "session_id" in blob
    assert "never invent" in blob
    assert "run" in blob  # execute only with RUN


def test_operator_manual_takes_no_arguments():
    # No path/argument surface -> cannot be coerced into arbitrary file reads.
    sig = inspect.signature(zeto.get_zeto_operator_manual)
    assert len(sig.parameters) == 0


def test_operator_manual_is_read_only():
    # Calling it must not touch the Research API at all (no LLM, no execution,
    # no session creation, no workflow mutation, no artefact inspection).
    src = inspect.getsource(zeto.get_zeto_operator_manual)
    assert "_api" not in src
    # Returns only the fixed manual path — no arbitrary path is echoed back.
    out = zeto.get_zeto_operator_manual()
    assert out["data"]["manual_path"] == "docs/LM_STUDIO_QWEN_OPERATOR_MANUAL.md"


# ---------------------------------------------------------------------------
# Visible-state contract on EVERY tool
# ---------------------------------------------------------------------------


def _call_every_tool() -> dict[str, dict]:
    """Invoke each tool once (Research API mocked) and return name -> response."""
    with ExitStack() as stack:
        p = lambda name, **kw: stack.enter_context(patch(f"{_MOD}.{name}", **kw))  # noqa: E731
        p("list_all_experiments", return_value=["exp_a"])
        p("create_research_session", return_value=_stub_session())
        p("load_research_session", return_value=_stub_session())
        p("summarize_research_session", return_value=_stub_summary())
        p("record_session_event", side_effect=lambda session, **kw: session)
        p("build_llm_context", return_value=_stub_ctx())
        p("compute_context_hash", return_value="a" * 64)
        p("run_llm_review", return_value=_stub_review())
        p("generate_iteration_proposal", return_value=_stub_proposal())
        p("generate_experiment_draft", return_value=_stub_draft(False))
        p("load_experiment_draft", return_value=_stub_draft(True))
        p("validate_experiment_draft", return_value=DraftValidationResult(is_valid=True, errors=[]))
        p("approve_experiment_draft", return_value=_stub_draft(True))
        p("render_draft_to_yaml", return_value="name: exp_a_v2\n# hash")
        p("execute_approved_config", return_value=_stub_execution(True))
        p("get_research_workflow_state", return_value=_stub_state())
        p("list_research_sessions", return_value=["s1", "s2"])
        p("get_latest_research_session", return_value=_stub_session())
        return {
            "get_zeto_operator_manual": zeto.get_zeto_operator_manual(),
            "list_experiments": zeto.list_experiments(),
            "create_research_session": zeto.create_research_session("exp_a", "goal"),
            "get_session_summary": zeto.get_session_summary("s1"),
            "list_research_sessions": zeto.list_research_sessions(),
            "get_latest_research_session": zeto.get_latest_research_session(),
            "check_research_workflow_state": zeto.check_research_workflow_state("exp_a"),
            "build_context_summary": zeto.build_context_summary("exp_a"),
            "run_experiment_review": zeto.run_experiment_review("exp_a"),
            "generate_iteration_proposal": zeto.generate_iteration_proposal("exp_a"),
            "generate_experiment_draft": zeto.generate_experiment_draft("exp_a"),
            "validate_experiment_draft": zeto.validate_experiment_draft("exp_a", "d1"),
            "approve_experiment_draft": zeto.approve_experiment_draft("exp_a", "d1"),
            "render_draft_to_yaml": zeto.render_draft_to_yaml("exp_a", "d1"),
            "execute_approved_config": zeto.execute_approved_config(
                "configs/experiments/exp_a_v2.yaml", confirmation="RUN"
            ),
            "review_post_run_result": zeto.review_post_run_result("exp_a"),
        }


def test_every_tool_returns_contract_envelope():
    for name, out in _call_every_tool().items():
        assert _CONTRACT_KEYS.issubset(out), f"{name} missing contract keys: {_CONTRACT_KEYS - set(out)}"
        assert isinstance(out["ok"], bool), name
        assert isinstance(out["stage"], str) and out["stage"], name
        assert isinstance(out["display"], str) and out["display"].strip(), name
        assert isinstance(out["data"], dict), name
        assert isinstance(out["next_suggested_action"], str) and out["next_suggested_action"], name


def test_error_responses_also_carry_contract():
    with patch(f"{_MOD}.load_research_session", return_value=None):
        out = zeto.get_session_summary("missing")
    assert _CONTRACT_KEYS.issubset(out)
    assert out["ok"] is False
    assert "error" in out["data"]


# ---------------------------------------------------------------------------
# Session-id guidance + recovery tools
# ---------------------------------------------------------------------------


def test_get_session_summary_not_found_points_to_recovery():
    # Passing an experiment name (not a UUID) fails with guidance to recover.
    with patch(f"{_MOD}.load_research_session", return_value=None):
        out = zeto.get_session_summary("canonical_ml_showcase")
    assert out["ok"] is False
    assert "create_research_session" in out["display"]
    assert "experiment name" in out["display"].lower()
    assert out["next_suggested_action"] == "get_latest_research_session"


def test_list_research_sessions_wraps_api():
    with patch(f"{_MOD}.list_research_sessions", return_value=["s1", "s2"]) as m:
        out = zeto.list_research_sessions()
    m.assert_called_once()
    assert out["ok"] is True
    assert out["data"]["session_ids"] == ["s1", "s2"]


def test_get_latest_research_session_returns_uuid():
    with patch(f"{_MOD}.get_latest_research_session", return_value=_stub_session()), patch(
        f"{_MOD}.summarize_research_session", return_value=_stub_summary()
    ):
        out = zeto.get_latest_research_session()
    assert out["ok"] is True
    assert out["data"]["session_id"] == "s1"
    assert out["stage"] == "latest_session"


def test_get_latest_research_session_when_none():
    with patch(f"{_MOD}.get_latest_research_session", return_value=None):
        out = zeto.get_latest_research_session()
    assert out["ok"] is False
    assert out["data"]["session_id"] is None
    assert out["next_suggested_action"] == "create_research_session"


def test_session_aware_tool_descriptions_mention_session_id():
    session_aware = {
        "run_experiment_review",
        "generate_iteration_proposal",
        "generate_experiment_draft",
        "validate_experiment_draft",
        "approve_experiment_draft",
        "render_draft_to_yaml",
        "execute_approved_config",
        "review_post_run_result",
    }
    by_name = {t.name: t.description for t in zeto.mcp._tool_manager.list_tools()}
    for name in session_aware:
        desc = by_name[name]
        assert "session_id" in desc, f"{name} description omits session_id guidance"
        assert "create_research_session" in desc, f"{name} should name the UUID source"
    # get_session_summary must explicitly reject experiment names.
    assert "experiment name" in by_name["get_session_summary"].lower()


# ---------------------------------------------------------------------------
# check_research_workflow_state (read-only preflight)
# ---------------------------------------------------------------------------


def _state_check(state: dict) -> dict:
    with patch(f"{_MOD}.get_research_workflow_state", return_value=state) as m:
        out = zeto.check_research_workflow_state("exp_a")
    m.assert_called_once_with("exp_a")
    return out


_WORKFLOW_STATE_KEYS = {
    "experiment_name",
    "baseline_artefacts_exist",
    "context_ready",
    "review_exists",
    "proposal_exists",
    "draft_exists",
    "latest_draft_id",
    "latest_draft_approved",
    "rendered_yaml_exists",
    "rendered_yaml_path",
    "revised_artefacts_exist",
    "report_path",
    "plots_dir",
}


def test_workflow_state_is_read_only_envelope():
    out = _state_check(_stub_state())
    assert _CONTRACT_KEYS.issubset(out)
    assert out["ok"] is True
    assert out["stage"] == "workflow_state_checked"
    assert out["data"]["experiment_name"] == "exp_a"


def test_workflow_state_data_is_compact_subset_only():
    # Only the compact subset is returned — no full filesystem inspection.
    out = _state_check(_stub_state())
    assert set(out["data"]) == _WORKFLOW_STATE_KEYS
    # Verbose / internal-only filesystem details must not leak.
    for dropped in ("metadata_exists", "metrics_exists", "proposed_name"):
        assert dropped not in out["data"]
    # Comfortably small for the chat window.
    assert len(json.dumps(out)) < 4000


def test_workflow_state_next_actions():
    # Each state maps to the correct next step in the pipeline order.
    cases = [
        (_stub_state(context_ready=False, baseline_artefacts_exist=False), "execute_approved_config"),
        (_stub_state(review_exists=False), "run_experiment_review"),
        (_stub_state(review_exists=True, proposal_exists=False), "generate_iteration_proposal"),
        (_stub_state(review_exists=True, proposal_exists=True, draft_exists=False), "generate_experiment_draft"),
        (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                     latest_draft_id="d1", latest_draft_approved=False), "approve_experiment_draft"),
        (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                     latest_draft_id="d1", latest_draft_approved=True, proposed_name="exp_a_v2",
                     rendered_yaml_exists=False), "render_draft_to_yaml"),
        (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                     latest_draft_id="d1", latest_draft_approved=True, proposed_name="exp_a_v2",
                     rendered_yaml_exists=True, revised_artefacts_exist=False), "execute_approved_config"),
        (_stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                     latest_draft_id="d1", latest_draft_approved=True, proposed_name="exp_a_v2",
                     rendered_yaml_exists=True, revised_artefacts_exist=True), "review_post_run_result"),
    ]
    for state, expected_next in cases:
        out = _state_check(state)
        assert out["next_suggested_action"] == expected_next, out["display"]


def test_workflow_state_display_example():
    out = _state_check(
        _stub_state(review_exists=True, proposal_exists=True, draft_exists=False)
    )
    # Concise, mentions what exists/missing and the next action.
    assert "draft missing" in out["display"]
    assert "generate_experiment_draft" in out["display"]


def test_workflow_state_unapproved_draft_mentions_validate():
    out = _state_check(
        _stub_state(review_exists=True, proposal_exists=True, draft_exists=True,
                    latest_draft_id="d1", latest_draft_approved=False)
    )
    assert out["next_suggested_action"] == "approve_experiment_draft"
    assert "validate" in out["display"].lower()


# ---------------------------------------------------------------------------
# Required display content for specific tools
# ---------------------------------------------------------------------------


def test_build_context_summary_display_shows_failure_modes_and_validation():
    with (
        patch(f"{_MOD}.build_llm_context", return_value=_stub_ctx()),
        patch(f"{_MOD}.compute_context_hash", return_value="h" * 64),
    ):
        out = zeto.build_context_summary("exp_a")
    assert out["stage"] == "context_built"
    # Failure modes visible in display.
    assert "poor_oos_consistency" in out["display"]
    # Key validation metric visible in display.
    assert "mean_oos_sharpe" in out["display"]
    assert out["data"]["failure_modes"][0]["name"] == "poor_oos_consistency"


def test_generate_draft_display_shows_config_diff():
    with patch(f"{_MOD}.generate_experiment_draft", return_value=_stub_draft(False)):
        out = zeto.generate_experiment_draft("exp_a")
    assert out["data"]["approved"] is False
    # The proposed config diff is human-visible.
    assert "model.params.alpha: 0.5 -> 1.0" in out["display"]


def test_validate_display_shows_pass_and_not_blocked():
    with patch(f"{_MOD}.load_experiment_draft", return_value=_stub_draft(False)), patch(
        f"{_MOD}.validate_experiment_draft", return_value=DraftValidationResult(True, [])
    ):
        out = zeto.validate_experiment_draft("exp_a", "d1")
    assert out["ok"] is True
    assert "PASS" in out["display"]
    assert out["data"]["rendering_blocked"] is False


def test_validate_display_shows_fail_and_blocked():
    bad = DraftValidationResult(is_valid=False, errors=["alpha out of range"])
    with patch(f"{_MOD}.load_experiment_draft", return_value=_stub_draft(False)), patch(
        f"{_MOD}.validate_experiment_draft", return_value=bad
    ):
        out = zeto.validate_experiment_draft("exp_a", "d1")
    assert out["ok"] is False
    assert "FAIL" in out["display"] and "BLOCK" in out["display"].upper()
    assert out["data"]["rendering_blocked"] is True


def test_render_display_states_execution_not_occurred():
    with (
        patch(f"{_MOD}.load_experiment_draft", return_value=_stub_draft(True)),
        patch(f"{_MOD}.render_draft_to_yaml", return_value="name: exp_a_v2\n# hash"),
    ):
        out = zeto.render_draft_to_yaml("exp_a", "d1")
    assert out["stage"] == "yaml_rendered"
    assert "NOT occurred" in out["display"]
    assert out["data"]["config_path"].endswith("exp_a_v2.yaml")


def test_review_post_run_display_shows_hash_flags_sections():
    with (
        patch(f"{_MOD}.run_llm_review", return_value=_stub_review()),
        patch(f"{_MOD}.build_llm_context", return_value=_stub_ctx()),
        patch(f"{_MOD}.compute_context_hash", return_value="abc12345" + "0" * 56),
    ):
        out = zeto.review_post_run_result("exp_a_v2")
    assert out["stage"] == "post_run_review_generated"
    assert "abc12345" in out["display"]               # post-run context hash
    assert "poor_oos_consistency" in out["display"]   # flag
    assert "performance_interpretation" in out["display"]  # review section
    assert out["data"]["context_hash"].startswith("abc12345")
    assert out["data"]["flags"]
    assert out["data"]["section_names"]
    assert "review_path" in out["data"]
    # Compact: no full review section bodies.
    assert "sections" not in out["data"]


# ---------------------------------------------------------------------------
# Delegation + governance
# ---------------------------------------------------------------------------


def test_list_experiments_wraps_api():
    with patch(f"{_MOD}.list_all_experiments", return_value=["a", "b"]) as m:
        out = zeto.list_experiments()
    m.assert_called_once()
    assert out["data"] == {"experiments": ["a", "b"]}


def test_create_session_wraps_api():
    with patch(f"{_MOD}.create_research_session", return_value=_stub_session()) as m:
        out = zeto.create_research_session("exp_a", "goal")
    m.assert_called_once()
    assert out["data"]["session_id"] == "s1"
    assert out["data"]["status"] == "active"


_STUB_CONFIG = {
    "version": "2",
    "name": "exp_a",
    "universe": {"tickers": ["SPY"]},
    "date_range": {"start": "2013-01-01", "end": "2024-12-31"},
    "model": {"type": "RidgeRegression", "params": {"alpha": 0.5}},
    "labels": {"type": "forward_returns", "params": {"horizon": 21}},
    "signal": {"type": "sign", "params": {}},
    "validation": {
        "type": "rolling",
        "parameters": {"train_months": 48, "test_months": 12, "gap_days": 0},
    },
    "execution": {"transaction_cost_bps": 5},
    "portfolio_construction": {
        "weighting": {
            "scheme": "equal_weight",
            "prediction_normalization": "none",
            "temperature": None,
        }
    },
    "features": {
        "ticker": "SPY",
        "entries": [{"name": "mom_20", "type": "momentum", "params": {"lookback": 20}}],
    },
}

_STUB_PROPOSAL = {
    "experiment_name": "exp_a",
    "generated_at": "2026-05-30T00:00:00+00:00",
    "context_hash": "c" * 64,
    "research_focus": "reduce split variance",
    "rationale": "high std OOS sharpe",
    "supporting_evidence": ["std_oos_sharpe=0.8"],
    "suggested_experiments": ["stronger regularisation"],
    "instability_signals": [],
    "validation_concerns": ["catastrophic split"],
    "feature_risks": [],
    "confidence": "medium",
}


def test_generate_draft_stub_returns_valid_envelope_with_diff(tmp_path):
    """provider='stub' must yield a valid draft envelope with a visible diff.

    First runs the REAL Research API stub path (isolated tmp dirs) — proving the
    stub no longer fails on invalid JSON — then wraps that real draft through the
    MCP tool to confirm the envelope + visible config diff.
    """
    from src.orchestration.api import research_api as real_api

    cfg_dir = tmp_path / "configs" / "experiments"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "exp_a.yaml").write_text(yaml.dump(_STUB_CONFIG), encoding="utf-8")
    llm_dir = tmp_path / "llm"
    (llm_dir / "exp_a").mkdir(parents=True)
    (llm_dir / "exp_a" / "iteration_proposal.json").write_text(
        json.dumps(_STUB_PROPOSAL), encoding="utf-8"
    )

    # Real stub generation (no call_llm mock): proves the stub yields a draft.
    real_draft = real_api.generate_experiment_draft(
        "exp_a", provider="stub", configs_base=cfg_dir, llm_base=llm_dir
    )
    assert real_draft.proposed_name == "exp_a_v2"
    assert real_draft.changes[0].current_value == 0.5

    # Wrap that real draft through the MCP tool to check the envelope + diff.
    with patch(f"{_MOD}.generate_experiment_draft", return_value=real_draft):
        out = zeto.generate_experiment_draft("exp_a", provider="stub")

    assert _CONTRACT_KEYS.issubset(out)
    assert out["ok"] is True
    assert out["stage"] == "draft_generated"
    assert out["data"]["approved"] is False
    assert out["data"]["proposed_name"] == "exp_a_v2"
    # Visible config diff in the human-readable display.
    assert "model.params.alpha: 0.5 -> 1.0" in out["display"]
    change = out["data"]["diff"][0]
    assert change["field"] == "model.params.alpha"
    assert change["current_value"] == 0.5
    assert change["proposed_value"] == 1.0
    assert "draft_path" in out["data"]
    # Compact: no full base config or full draft object.
    assert "changes" not in out["data"]


def test_generate_draft_failure_returns_clean_envelope():
    # A failing draft generation (e.g. missing proposal) must NOT raise or return
    # a partial/invented config — it must surface a clean ok=False envelope so the
    # model reports the failure and stops.
    with patch(f"{_MOD}.generate_experiment_draft", side_effect=FileNotFoundError("no proposal")):
        out = zeto.generate_experiment_draft("exp_a", provider="stub")
    assert _CONTRACT_KEYS.issubset(out)
    assert out["ok"] is False
    assert out["stage"] == "draft_generation_failed"
    assert "no proposal" in out["data"]["error"]
    assert "changes" not in out["data"]


def test_generate_draft_does_not_auto_approve():
    with (
        patch(f"{_MOD}.generate_experiment_draft", return_value=_stub_draft(False)),
        patch(f"{_MOD}.approve_experiment_draft") as m_approve,
    ):
        out = zeto.generate_experiment_draft("exp_a")
    assert out["data"]["approved"] is False
    m_approve.assert_not_called()


def test_render_refuses_unapproved_draft():
    with (
        patch(f"{_MOD}.load_experiment_draft", return_value=_stub_draft(approved=False)),
        patch(f"{_MOD}.render_draft_to_yaml") as m_render,
    ):
        out = zeto.render_draft_to_yaml("exp_a", "d1")
    assert out["ok"] is False
    assert out["stage"] == "render_blocked"
    m_render.assert_not_called()


def test_render_approved_draft_delegates():
    with (
        patch(f"{_MOD}.load_experiment_draft", return_value=_stub_draft(approved=True)),
        patch(f"{_MOD}.render_draft_to_yaml", return_value="name: exp_a_v2\n# hash") as m_render,
    ):
        out = zeto.render_draft_to_yaml("exp_a", "d1")
    m_render.assert_called_once()
    assert out["data"]["config_path"] == "configs/experiments/exp_a_v2.yaml"
    assert out["data"]["execution_has_occurred"] is False
    # Compact: YAML content is not returned.
    assert "yaml_preview" not in out["data"]
    assert "yaml" not in out["data"]


def test_approve_tool_delegates_once():
    with (
        patch(f"{_MOD}.load_experiment_draft", return_value=_stub_draft(approved=False)),
        patch(f"{_MOD}.approve_experiment_draft", return_value=_stub_draft(approved=True)) as m,
    ):
        out = zeto.approve_experiment_draft("exp_a", "d1")
    m.assert_called_once()
    assert out["data"]["approved"] is True


# ---------------------------------------------------------------------------
# Execution requires confirmation='RUN'
# ---------------------------------------------------------------------------


def test_execute_refuses_missing_confirmation():
    with patch(f"{_MOD}.execute_approved_config") as m:
        out = zeto.execute_approved_config("configs/experiments/x.yaml")
    assert out["ok"] is False
    assert out["stage"] == "execution_refused"
    assert "RUN" in out["data"]["error"]
    m.assert_not_called()


def test_execute_refuses_wrong_confirmation():
    with patch(f"{_MOD}.execute_approved_config") as m:
        out = zeto.execute_approved_config("configs/experiments/x.yaml", confirmation="run")
    assert out["ok"] is False
    assert out["data"]["success"] is False
    m.assert_not_called()


def test_execute_with_run_delegates_exactly_once():
    with patch(f"{_MOD}.execute_approved_config", return_value=_stub_execution(True)) as m:
        out = zeto.execute_approved_config("configs/experiments/x.yaml", confirmation="RUN")
    m.assert_called_once()
    assert out["ok"] is True
    assert out["data"]["success"] is True
    assert out["data"]["experiment_name"] == "exp_a_v2"
    assert out["data"]["plots_dir"].endswith("/plots")
    # Compact: no full run object / report content.
    assert "execution" not in out["data"]


def test_execute_dry_run_plans_only():
    planned = _stub_execution(True)
    planned.experiment_name = None
    planned.artefact_root = None
    with (
        patch(f"{_MOD}.execute_approved_config", return_value=planned) as m,
        patch(f"{_MOD}.record_session_event") as m_event,
    ):
        out = zeto.execute_approved_config(
            "configs/experiments/x.yaml", confirmation="RUN", dry_run=True
        )
    assert out["stage"] == "execution_planned"
    assert out["data"]["planned"] is True
    m_event.assert_not_called()
    assert m.call_args.kwargs["dry_run"] is True


# ---------------------------------------------------------------------------
# Compactness guards (keep responses inside LM Studio's ~32k context window)
# ---------------------------------------------------------------------------

# Verbose payloads that must never be inlined into LM Studio chat.
_FORBIDDEN_DATA_KEYS = {
    "performance", "validation", "sections", "session_summary", "execution",
    "changes", "yaml", "yaml_preview", "rationale", "supporting_evidence",
    "suggested_experiments", "per_split_sharpes", "context", "review",
}


def test_no_verbose_keys_in_any_tool_data():
    for name, out in _call_every_tool().items():
        leaked = set(out["data"]) & _FORBIDDEN_DATA_KEYS
        assert not leaked, f"{name} leaks verbose keys: {leaked}"


def test_all_tool_payloads_are_small():
    # Generous per-tool ceilings; the principle is "small for a 32k window".
    bigger = {"execute_approved_config": 6000, "get_session_summary": 6000}
    for name, out in _call_every_tool().items():
        size = len(json.dumps(out))
        assert size < bigger.get(name, 4000), f"{name} payload too large: {size}"


def test_context_summary_drops_full_dicts_and_arrays():
    ctx = SimpleNamespace(
        experiment_name="exp_a",
        failure_modes=[{"name": "x", "severity": "critical", "description": "D" * 5000}],
        performance={"sharpe_ratio": 0.7, "max_drawdown_pct": "-34.10%", "junk": "Y" * 5000},
        validation={
            "mean_oos_sharpe": -0.3, "std_oos_sharpe": 1.2, "n_splits": 7,
            "n_negative_sharpe_splits": 4, "per_split_sharpes": [0.1] * 500,
        },
    )
    with (
        patch(f"{_MOD}.build_llm_context", return_value=ctx),
        patch(f"{_MOD}.compute_context_hash", return_value="h" * 64),
    ):
        out = zeto.build_context_summary("exp_a")
    blob = json.dumps(out)
    assert "performance" not in out["data"] and "validation" not in out["data"]
    assert "per_split_sharpes" not in blob
    assert "Y" * 5000 not in blob          # raw perf junk dropped
    assert "D" * 5000 not in blob          # failure-mode descriptions dropped
    assert out["data"]["key_metrics"]["sharpe_ratio"] == 0.7
    assert len(blob) < 4000


def test_review_excludes_full_section_bodies():
    big = "X" * 50000
    review = SimpleNamespace(
        sections={"performance_assessment": big, "validation_robustness": big},
        flags=[],
    )
    with (
        patch(f"{_MOD}.run_llm_review", return_value=review),
        patch(f"{_MOD}.build_llm_context", return_value=_stub_ctx()),
        patch(f"{_MOD}.compute_context_hash", return_value="h" * 64),
    ):
        out = zeto.run_experiment_review("exp_a")
    blob = json.dumps(out)
    assert big not in blob                  # full bodies not returned
    assert out["data"]["section_names"] == ["performance_assessment", "validation_robustness"]
    assert "review_path" in out["data"]
    assert len(blob) < 4000


def test_proposal_excludes_full_bodies():
    proposal = SimpleNamespace(
        research_focus="short focus",
        rationale="R" * 5000,
        supporting_evidence=["E" * 5000] * 10,
        suggested_experiments=["S" * 5000] * 10,
        validation_concerns=["c1", "c2", "c3", "c4", "c5"],
        confidence="medium",
        context_hash="h" * 64,
    )
    with patch(f"{_MOD}.generate_iteration_proposal", return_value=proposal):
        out = zeto.generate_iteration_proposal("exp_a")
    blob = json.dumps(out)
    assert "R" * 5000 not in blob
    assert "rationale" not in out["data"]
    assert len(out["data"]["validation_concerns"]) <= 3
    assert "proposal_path" in out["data"]
    assert len(blob) < 4000


def test_context_hash_preserved_full_in_data_short_in_display():
    full_hash = "83765162d7d91cadbfae8450cb1da3bdcb4c23e721253eff9587a885ff451b99"
    with (
        patch(f"{_MOD}.build_llm_context", return_value=_stub_ctx()),
        patch(f"{_MOD}.compute_context_hash", return_value=full_hash),
    ):
        out = zeto.build_context_summary("exp_a")
    assert out["data"]["context_hash"] == full_hash          # full provenance kept
    assert full_hash[:8] in out["display"]                   # short in display
    assert full_hash not in out["display"]                   # not the full hash


# ---------------------------------------------------------------------------
# Non-coupling
# ---------------------------------------------------------------------------


def test_module_has_no_forbidden_imports_or_calls():
    src = inspect.getsource(zeto)
    for forbidden in [
        "import subprocess",
        "subprocess.",
        "os.system",
        "run_from_config",
        "eval(",
        "exec(",
        "from src.experiments",
        "import src.experiments",
        "from src.backtesting",
        "from src.portfolio",
        "from src.data",
        "from src.models",
        "from src.features",
    ]:
        assert forbidden not in src, f"zeto_server must not use {forbidden!r}"


def test_module_imports_only_research_api_surface():
    import_lines = [
        line
        for line in inspect.getsource(zeto).splitlines()
        if line.startswith("from src.") or line.startswith("import src.")
    ]
    for line in import_lines:
        assert (
            "research_api" in line or "session_schema" in line
        ), f"unexpected orchestration import: {line}"
