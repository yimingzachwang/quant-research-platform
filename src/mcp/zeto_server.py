"""Zeto MCP server — exposes the governed Research API as MCP tools.

Runs over stdio so an MCP client (e.g. LM Studio) can drive the AI-assisted
research workflow:

    list experiments -> create session -> build context -> review -> proposal
    -> draft -> validate -> approve -> render YAML
    -> (optional, explicitly confirmed) execute -> post-run review -> summary

Governance held by this layer:
  * tools wrap ``src.orchestration.api.research_api`` ONLY — no quant engine
    internals, no subprocess, no shell, no file/eval access;
  * draft approval is its own explicit tool and is never performed implicitly;
  * YAML rendering is separate from execution;
  * execution requires ``confirmation="RUN"`` and runs exactly one config —
    no loop, no retry, no automatic lineage registration.

The quant engine remains authoritative; these tools coordinate and interpret.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src.orchestration.api import research_api as _api  # noqa: E402
from src.orchestration.session.session_schema import SessionEventType  # noqa: E402

mcp = FastMCP("zeto")


# ---------------------------------------------------------------------------
# Internal helpers (not exposed as tools)
# ---------------------------------------------------------------------------


def _record_event(
    session_id: str | None,
    event_type: str,
    experiment_name: str,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    """Record one session event if a valid session_id is given; else no-op.

    Returns the session summary (or None when there is no session to record on).
    """
    if not session_id:
        return None
    session = _api.load_research_session(session_id)
    if session is None:
        return None
    session = _api.record_session_event(
        session=session,
        event_type=event_type,
        experiment_name=experiment_name,
        data=data,
    )
    return _api.summarize_research_session(session)


def _short_hash(h: str | None, n: int = 8) -> str:
    """First n characters of a hash for display (full hash stays in data)."""
    return (h or "")[:n]


def _short(text: str, limit: int = 200) -> str:
    """Truncate a string so a compact payload never grows unbounded."""
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _name_sev(failure_modes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact failure-mode/flag form: only name + severity (no descriptions)."""
    return [
        {"name": fm.get("name"), "severity": fm.get("severity")}
        for fm in (failure_modes or [])
    ]


def _compact_diff(changes: list[Any]) -> list[dict[str, Any]]:
    """Compact config diff: dotted field + current/proposed values only."""
    return [
        {
            "field": f"{c.section}.{c.field}",
            "current_value": c.current_value,
            "proposed_value": c.proposed_value,
        }
        for c in changes
    ]


def _envelope(
    ok: bool,
    stage: str,
    display: str,
    data: dict[str, Any],
    next_suggested_action: str,
) -> dict[str, Any]:
    """The visible-state contract returned by every tool.

    ok:                    whether the tool succeeded (False for errors/refusals)
    stage:                 machine-readable lifecycle stage
    display:               concise human-readable line for LM Studio to show
    data:                  the structured payload
    next_suggested_action: the single recommended next tool (advisory; the
                           researcher/model decides — there is no auto-chaining)
    """
    return {
        "ok": ok,
        "stage": stage,
        "display": display,
        "data": data,
        "next_suggested_action": next_suggested_action,
    }


def _fmt_failure_modes(failure_modes: list[dict[str, Any]]) -> str:
    if not failure_modes:
        return "none"
    return "; ".join(
        f"[{(fm.get('severity') or '?').upper()}] {fm.get('name')}" for fm in failure_modes
    )


def _fmt_validation(validation: dict[str, Any] | None) -> str:
    val = validation or {}
    keys = [
        "mean_oos_sharpe",
        "std_oos_sharpe",
        "n_splits",
        "n_negative_sharpe_splits",
        "worst_split_sharpe",
    ]
    shown = [f"{k}={val[k]}" for k in keys if k in val and val[k] is not None]
    return ", ".join(shown) if shown else "n/a"


def _review_experiment(
    experiment_name: str,
    session_id: str | None,
    provider: str,
    model: str | None,
    base_url: str | None,
    event_type: str,
    stage: str,
    label: str,
    next_action: str,
) -> dict[str, Any]:
    """Shared LLM-review path for pre-run and post-run reviews.

    Compact: returns context_hash, short flags (name+severity), section NAMES,
    and the path to the persisted review.  Full section bodies stay on disk.
    """
    review = _api.run_llm_review(
        experiment_name,
        provider=provider,
        model=model,
        base_url=base_url,
        persist_context=False,
        persist_review=True,
    )
    ctx = _api.build_llm_context(experiment_name, persist=False)
    context_hash = _api.compute_context_hash(ctx)
    _record_event(
        session_id,
        event_type,
        experiment_name,
        {"provider": provider, "context_hash": context_hash},
    )
    flags = _name_sev(ctx.failure_modes)
    section_names = list(review.sections.keys())
    display = (
        f"{label} of '{experiment_name}' (hash {_short_hash(context_hash)}). "
        f"Flags: {_fmt_failure_modes(flags)}. "
        f"Sections: {', '.join(section_names) or '(none parsed)'}."
    )
    data = {
        "experiment_name": experiment_name,
        "context_hash": context_hash,
        "flags": flags,
        "section_names": section_names,
        "review_path": _api.review_artifact_path(experiment_name),
    }
    return _envelope(True, stage, display, data, next_action)


# ---------------------------------------------------------------------------
# Operator manual (fixed, read-only)
# ---------------------------------------------------------------------------

_OPERATOR_MANUAL_PATH = "docs/LM_STUDIO_QWEN_OPERATOR_MANUAL.md"

# Fixed, in-code rules — returned as-is. This tool never reads files.
_OPERATOR_RULES: tuple[str, ...] = (
    "Use exact snake_case MCP tool names; do not camelCase tool names.",
    "Show each tool's display field after every tool call.",
    "Use compact tool outputs only; reason from paths, hashes, and IDs.",
    "Before major transitions, call check_research_workflow_state and follow its next_suggested_action.",
    "Store the session_id UUID returned by create_research_session.",
    "Pass session_id to all session-aware tools (review, proposal, draft, "
    "validate, approve, render, execute, post-run review, get_session_summary).",
    "If session_id is lost, call get_latest_research_session to recover it.",
    "Never use experiment_name, context_hash, draft_id, or config_path as session_id.",
    "Never invent metrics; use tool outputs only.",
    "Never invent or hand-write draft configs; drafts come from generate_experiment_draft only.",
    "If a tool returns ok=false, report the failure verbatim and stop.",
    "Do not auto-approve drafts; ask the user before approval.",
    "Do not render YAML unless the user approves the draft.",
    "Do not execute unless the user explicitly provides RUN (confirmation='RUN').",
    "Do not run extra experiments, optimise automatically, or loop.",
    "For local Qwen-backed review/proposal/draft, use provider=openai, model=qwen2.5-7b-instruct, base_url=http://127.0.0.1:1234/v1.",
    "Governed sequence: create session -> check state -> execute baseline if "
    "needed -> build context -> review -> proposal -> draft -> validate -> "
    "approval -> render YAML -> RUN execution -> post-run review -> session summary.",
    "The quant engine remains authoritative; Qwen coordinates and interprets only.",
)


def get_zeto_operator_manual() -> dict[str, Any]:
    """Return the fixed Zeto operator manual (read-only).

    Loads a compact rules summary so an LM Studio chat does not need long rules
    pasted in. This tool is read-only: it executes nothing, calls no LLM,
    inspects no experiment artefacts, creates no session, mutates no state, and
    reads no files — it returns only the fixed in-code manual (no arbitrary
    paths).
    """
    display = (
        "Zeto operator manual loaded. Use exact snake_case tools, check workflow "
        "state before transitions, preserve session_id, never invent configs, "
        "and execute only with RUN."
    )
    data = {
        "manual_path": _OPERATOR_MANUAL_PATH,
        "rules": list(_OPERATOR_RULES),
    }
    return _envelope(
        True, "operator_manual_loaded", display, data, "check_research_workflow_state"
    )


# ---------------------------------------------------------------------------
# Phase 1 tools
# ---------------------------------------------------------------------------


def list_experiments() -> dict[str, Any]:
    """List experiments that have persisted result artefacts on disk."""
    experiments = _api.list_all_experiments()
    preview = ", ".join(experiments[:8]) + ("..." if len(experiments) > 8 else "")
    display = f"{len(experiments)} experiment(s) available: {preview or '(none)'}"
    return _envelope(
        True, "experiments_listed", display, {"experiments": experiments},
        "create_research_session",
    )


def create_research_session(root_experiment: str, research_goal: str) -> dict[str, Any]:
    """Create a research session anchored to one experiment and a stated goal."""
    session = _api.create_research_session(
        root_experiment=root_experiment,
        research_goal=research_goal,
    )
    data = {
        "session_id": session.session_id,
        "research_goal": session.research_goal,
        "root_experiment": session.root_experiment,
        "active_experiment": session.active_experiment,
        "status": session.status,
    }
    display = (
        f"Session {session.session_id} created for '{session.root_experiment}' "
        f"· goal: {session.research_goal}"
    )
    return _envelope(True, "session_created", display, data, "build_context_summary")


def get_session_summary(session_id: str) -> dict[str, Any]:
    """Return the current state of a research session (events, draft, config).

    session_id MUST be the UUID returned by create_research_session — NOT an
    experiment name, context_hash, or draft_id.  If the UUID was lost, call
    get_latest_research_session (or list_research_sessions) to recover it.
    """
    session = _api.load_research_session(session_id)
    if session is None:
        return _envelope(
            False, "session_not_found",
            f"Session '{session_id}' not found. session_id must be the UUID from "
            "create_research_session (not an experiment name / hash / draft id). "
            "Call get_latest_research_session to recover the active session.",
            {"error": f"Session '{session_id}' not found"},
            "get_latest_research_session",
        )
    summary = _api.summarize_research_session(session)
    active_draft = summary.get("active_draft")
    active_draft_id = (
        active_draft.get("draft_id") if isinstance(active_draft, dict) else active_draft
    )
    data = {
        "session_id": summary["session_id"],
        "research_goal": _short(summary["research_goal"]),
        "event_count": summary["event_count"],
        "active_experiment": summary["active_experiment"],
        "active_draft": active_draft_id,
        "approved_config_path": summary["approved_config_path"],
        "experiments_visited": summary["experiments_visited"],
        "status": summary["status"],
    }
    display = (
        f"Session {summary['session_id']}: {summary['event_count']} event(s); "
        f"active draft: {active_draft_id or 'none'}; "
        f"approved config: {summary['approved_config_path'] or 'none'}"
    )
    return _envelope(True, "session_summary", display, data, "run_experiment_review")


def list_research_sessions() -> dict[str, Any]:
    """List existing research session IDs (UUIDs). Read-only.

    Use this to recover a session_id you lost. Pass one of these UUIDs (not an
    experiment name) to the session-aware tools.
    """
    ids = _api.list_research_sessions()
    display = f"{len(ids)} research session(s): " + (", ".join(ids[:8]) or "(none)")
    return _envelope(
        True, "sessions_listed", display, {"session_ids": ids}, "get_latest_research_session",
    )


def get_latest_research_session() -> dict[str, Any]:
    """Return the most recently updated research session (UUID + compact summary).

    Read-only recovery helper for when the session_id was lost. The returned
    session_id is the UUID to pass to all session-aware tools — never use an
    experiment name, context_hash, or draft_id as a session_id.
    """
    session = _api.get_latest_research_session()
    if session is None:
        return _envelope(
            False, "no_sessions",
            "No research sessions found. Call create_research_session first.",
            {"session_id": None}, "create_research_session",
        )
    summary = _api.summarize_research_session(session)
    data = {
        "session_id": session.session_id,
        "research_goal": _short(session.research_goal),
        "active_experiment": session.active_experiment,
        "event_count": summary["event_count"],
        "status": session.status,
        "updated_at": session.updated_at,
    }
    display = (
        f"Latest session {session.session_id} (status: {session.status}, "
        f"{summary['event_count']} event(s)). Use this session_id for session-aware tools."
    )
    return _envelope(True, "latest_session", display, data, "check_research_workflow_state")


def _workflow_next_and_display(state: dict[str, Any]) -> tuple[str, str]:
    """Map a workflow-state dict to (next_suggested_action, concise display).

    Advisory only — names the single recommended next tool based on which
    artefacts already exist; it never runs anything.
    """
    name = state["experiment_name"]
    if not state["context_ready"]:
        nxt = "execute_approved_config"
        head = "baseline artefacts missing — execute the baseline config (confirmation='RUN')"
    elif not state["review_exists"]:
        nxt = "run_experiment_review"
        head = "baseline artefacts exist; review missing"
    elif not state["proposal_exists"]:
        nxt = "generate_iteration_proposal"
        head = "review exists; proposal missing"
    elif not state["draft_exists"]:
        nxt = "generate_experiment_draft"
        head = "proposal exists; draft missing"
    elif not state["latest_draft_approved"]:
        nxt = "approve_experiment_draft"
        head = f"draft {state['latest_draft_id']} exists (unapproved) — validate, then approve"
    elif not state["rendered_yaml_exists"]:
        nxt = "render_draft_to_yaml"
        head = "draft approved; YAML not yet rendered"
    elif not state["revised_artefacts_exist"]:
        nxt = "execute_approved_config"
        head = "YAML rendered; revised experiment not yet executed (confirmation='RUN')"
    else:
        nxt = "review_post_run_result"
        head = "revised artefacts exist; ready for post-run review"
    return nxt, f"[{name}] {head}. Next suggested action: {nxt}."


def check_research_workflow_state(experiment_name: str) -> dict[str, Any]:
    """Read-only preflight: report which research artefacts exist and the next step.

    Inspects on-disk state only (baseline artefacts incl. metadata.json/
    metrics.json, LLM review, iteration proposal, latest draft + approval,
    rendered YAML, revised run artefacts, report/plots paths).  Executes nothing,
    calls no LLM, and touches no quant-engine code.  Call this before major
    transitions so the workflow is never run out of order.
    """
    state = _api.get_research_workflow_state(experiment_name)
    next_action, display = _workflow_next_and_display(state)
    # Compact subset — flags/paths/ids only, no full filesystem inspection.
    data = {
        "experiment_name": state["experiment_name"],
        "baseline_artefacts_exist": state["baseline_artefacts_exist"],
        "context_ready": state["context_ready"],
        "review_exists": state["review_exists"],
        "proposal_exists": state["proposal_exists"],
        "draft_exists": state["draft_exists"],
        "latest_draft_id": state["latest_draft_id"],
        "latest_draft_approved": state["latest_draft_approved"],
        "rendered_yaml_exists": state["rendered_yaml_exists"],
        "rendered_yaml_path": state["rendered_yaml_path"],
        "revised_artefacts_exist": state["revised_artefacts_exist"],
        "report_path": state["report_path"],
        "plots_dir": state["plots_dir"],
    }
    return _envelope(True, "workflow_state_checked", display, data, next_action)


def build_context_summary(experiment_name: str) -> dict[str, Any]:
    """Assemble deterministic structured context for an experiment (no LLM call).

    Returns a compact view: context hash, detected failure modes, and the
    performance/validation blocks.  No raw tables or large payloads.
    """
    ctx = _api.build_llm_context(experiment_name, persist=False)
    context_hash = _api.compute_context_hash(ctx)
    failure_modes = _name_sev(ctx.failure_modes)
    perf = ctx.performance or {}
    val = ctx.validation or {}
    key_metrics = {
        "sharpe_ratio": perf.get("sharpe_ratio"),
        "max_drawdown_pct": perf.get("max_drawdown_pct"),
        "mean_oos_sharpe": val.get("mean_oos_sharpe"),
        "std_oos_sharpe": val.get("std_oos_sharpe"),
        "n_splits": val.get("n_splits"),
        "n_negative_sharpe_splits": val.get("n_negative_sharpe_splits"),
    }
    data = {
        "experiment_name": ctx.experiment_name,
        "context_hash": context_hash,
        "failure_modes": failure_modes,
        "key_metrics": key_metrics,
    }
    display = (
        f"Context for {ctx.experiment_name} built (hash {_short_hash(context_hash)}). "
        f"Failure modes: {_fmt_failure_modes(failure_modes)}. "
        f"Validation: {_fmt_validation(ctx.validation)}."
    )
    return _envelope(True, "context_built", display, data, "run_experiment_review")


def run_experiment_review(
    experiment_name: str,
    session_id: str | None = None,
    provider: str = "stub",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run an LLM review of an experiment's persisted diagnostics.

    Advisory only — interprets existing artefacts; computes nothing new and
    runs no experiment.  Records REVIEW_GENERATED when a session_id is given.
    """
    return _review_experiment(
        experiment_name,
        session_id,
        provider,
        model,
        base_url,
        SessionEventType.REVIEW_GENERATED,
        "review_generated",
        "Review",
        "generate_iteration_proposal",
    )


def generate_iteration_proposal(
    experiment_name: str,
    session_id: str | None = None,
    provider: str = "stub",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Generate an advisory next-step research proposal grounded in diagnostics.

    Does not prescribe parameter values and runs no experiment.  Records
    ITERATION_PROPOSAL_GENERATED when a session_id is given.
    """
    proposal = _api.generate_iteration_proposal(
        experiment_name,
        provider=provider,
        model=model,
        base_url=base_url,
        persist=True,
    )
    _record_event(
        session_id,
        SessionEventType.ITERATION_PROPOSAL_GENERATED,
        experiment_name,
        {"context_hash": proposal.context_hash, "research_focus": proposal.research_focus},
    )
    # Compact: short focus/confidence/path + at most 3 short concerns.
    concerns = [_short(c, 160) for c in (proposal.validation_concerns or [])[:3]]
    data = {
        "experiment_name": experiment_name,
        "context_hash": proposal.context_hash,
        "research_focus": _short(proposal.research_focus),
        "confidence": proposal.confidence,
        "validation_concerns": concerns,
        "proposal_path": _api.proposal_artifact_path(experiment_name),
    }
    display = (
        f"Proposal for {experiment_name} (hash {_short_hash(proposal.context_hash)}): "
        f"{_short(proposal.research_focus) or '(none)'} (confidence: {proposal.confidence or 'n/a'})"
    )
    return _envelope(True, "proposal_generated", display, data, "generate_experiment_draft")


def generate_experiment_draft(
    experiment_name: str,
    session_id: str | None = None,
    provider: str = "stub",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Synthesise a schema-bounded config draft (typed deltas, NOT YAML).

    Requires a prior persisted iteration proposal.  The draft is unapproved
    (approved=false) and must be validated and explicitly approved before it can
    be rendered.  Records DRAFT_GENERATED when a session_id is given.

    With provider='stub' the draft is deterministic (model.params.alpha -> 1.0)
    and no LLM is called.  On any failure this returns an ok=false envelope with
    the error — never a partial or invented draft.
    """
    try:
        draft = _api.generate_experiment_draft(
            experiment_name,
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean envelope, never a traceback
        return _envelope(
            False, "draft_generation_failed",
            f"Draft generation failed: {type(exc).__name__}: {exc}. "
            "Do not invent a config — report this and stop.",
            {"error": f"{type(exc).__name__}: {exc}"},
            "generate_iteration_proposal",
        )
    _record_event(
        session_id,
        SessionEventType.DRAFT_GENERATED,
        experiment_name,
        {
            "draft_id": draft.draft_id,
            "draft_hash": draft.draft_hash,
            "proposed_name": draft.proposed_name,
        },
    )
    diff = _compact_diff(draft.changes)
    diff_lines = [
        f"{d['field']}: {d['current_value']} -> {d['proposed_value']}" for d in diff
    ]
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft.draft_id,
        "proposed_name": draft.proposed_name,
        "approved": draft.approved,
        "diff": diff,
        "draft_path": _api.draft_artifact_path(experiment_name, draft.draft_id),
    }
    display = (
        f"Draft {draft.draft_id} '{draft.proposed_name}' (approved={draft.approved}). "
        f"Proposed config diff:\n  " + ("\n  ".join(diff_lines) or "(no changes)")
    )
    return _envelope(True, "draft_generated", display, data, "validate_experiment_draft")


def validate_experiment_draft(
    experiment_name: str,
    draft_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Validate a draft against the authoritative config schema (no side effects).

    Records DRAFT_VALIDATED when a session_id is given.
    """
    draft = _api.load_experiment_draft(experiment_name, draft_id)
    if draft is None:
        return _envelope(
            False, "draft_not_found",
            f"Draft '{draft_id}' not found for experiment '{experiment_name}'.",
            {"error": f"Draft '{draft_id}' not found for experiment '{experiment_name}'"},
            "generate_experiment_draft",
        )
    result = _api.validate_experiment_draft(draft)
    _record_event(
        session_id,
        SessionEventType.DRAFT_VALIDATED,
        experiment_name,
        {"draft_id": draft_id, "is_valid": result.is_valid, "error_count": len(result.errors)},
    )
    errors = [_short(e, 160) for e in result.errors[:3]]
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft_id,
        "is_valid": result.is_valid,
        "rendering_blocked": not result.is_valid,
        "error_count": len(result.errors),
        "errors": errors,
    }
    if result.is_valid:
        display = "Validation PASS — rendering is NOT blocked. Approval still required."
        next_action = "approve_experiment_draft"
    else:
        display = "Validation FAIL — rendering is BLOCKED. Errors: " + "; ".join(errors)
        next_action = "generate_experiment_draft"
    return _envelope(result.is_valid, "draft_validated", display, data, next_action)


def approve_experiment_draft(
    experiment_name: str,
    draft_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Explicitly approve a draft so it can be rendered to YAML.

    Approval is performed only by this tool — no other tool auto-approves.
    The approved draft is persisted.  Records DRAFT_APPROVED when a session_id
    is given.
    """
    draft = _api.load_experiment_draft(experiment_name, draft_id)
    if draft is None:
        return _envelope(
            False, "draft_not_found",
            f"Draft '{draft_id}' not found for experiment '{experiment_name}'.",
            {"error": f"Draft '{draft_id}' not found for experiment '{experiment_name}'"},
            "generate_experiment_draft",
        )
    approved = _api.approve_experiment_draft(draft)
    _record_event(
        session_id,
        SessionEventType.DRAFT_APPROVED,
        experiment_name,
        {"draft_id": approved.draft_id, "draft_hash": approved.draft_hash},
    )
    data = {
        "experiment_name": experiment_name,
        "draft_id": approved.draft_id,
        "approved": approved.approved,
        "approved_at": approved.approved_at,
        "draft_hash": approved.draft_hash,
    }
    display = f"Draft {approved.draft_id} approved at {approved.approved_at}."
    return _envelope(True, "draft_approved", display, data, "render_draft_to_yaml")


def render_draft_to_yaml(
    experiment_name: str,
    draft_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Render an APPROVED draft to a YAML config file. Does NOT execute it.

    Returns a clear error if the draft is not approved.  Execution requires a
    separate, explicitly confirmed tool call.  Records YAML_RENDERED when a
    session_id is given.
    """
    draft = _api.load_experiment_draft(experiment_name, draft_id)
    if draft is None:
        return _envelope(
            False, "draft_not_found",
            f"Draft '{draft_id}' not found for experiment '{experiment_name}'.",
            {"error": f"Draft '{draft_id}' not found for experiment '{experiment_name}'"},
            "generate_experiment_draft",
        )
    if not draft.approved:
        return _envelope(
            False, "render_blocked",
            f"Draft '{draft_id}' must be approved before rendering to YAML.",
            {"error": f"Draft '{draft_id}' must be approved before rendering to YAML"},
            "approve_experiment_draft",
        )

    # Render (and persist) the YAML on disk; do NOT return its contents.
    _api.render_draft_to_yaml(draft)
    config_path = f"configs/experiments/{draft.proposed_name}.yaml"
    _record_event(
        session_id,
        SessionEventType.YAML_RENDERED,
        experiment_name,
        {"draft_id": draft.draft_id, "config_path": config_path},
    )
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft.draft_id,
        "config_path": config_path,
        "draft_hash": draft.draft_hash,
        "execution_has_occurred": False,
    }
    display = (
        f"Rendered {config_path}. Execution has NOT occurred — call "
        "execute_approved_config with confirmation='RUN' to run it."
    )
    return _envelope(True, "yaml_rendered", display, data, "execute_approved_config")


# ---------------------------------------------------------------------------
# Phase 2 tools (optional execution)
# ---------------------------------------------------------------------------


def execute_approved_config(
    config_path: str,
    session_id: str | None = None,
    confirmation: str = "",
    report: bool = True,
    preset: str = "canonical",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run exactly one approved YAML config through the quant research engine.

    Requires confirmation='RUN'. This tool does not optimise, loop, retry,
    trade, or register lineage automatically. It runs a single config and
    returns the result. With dry_run=true it only reports the planned command
    and records nothing.
    """
    if confirmation != "RUN":
        return _envelope(
            False, "execution_refused",
            "Execution refused — confirmation='RUN' is required to run a config.",
            {"success": False, "error": "Execution requires confirmation='RUN'."},
            "execute_approved_config",
        )

    if dry_run:
        execution = _api.execute_approved_config(
            config_path, report=report, preset=preset, dry_run=True
        )
        data = {
            "success": False,
            "planned": True,
            "config_path": config_path,
            "command_hint": execution.command_hint,
        }
        display = f"Planned (dry run, NOT executed): {execution.command_hint}"
        return _envelope(True, "execution_planned", display, data, "execute_approved_config")

    _record_event(
        session_id,
        SessionEventType.EXECUTION_REQUESTED,
        config_path,
        {"config_path": config_path, "preset": preset, "report": report},
    )

    execution = _api.execute_approved_config(
        config_path, report=report, preset=preset, dry_run=False
    )

    if not execution.success:
        data = {"success": False, "config_path": config_path, "error": execution.error}
        return _envelope(
            False, "execution_failed",
            f"Execution failed: {execution.error}", data, "execute_approved_config",
        )

    _record_event(
        session_id,
        SessionEventType.EXECUTION_COMPLETED,
        execution.experiment_name or config_path,
        {
            "config_path": execution.config_path,
            "experiment_name": execution.experiment_name,
            "artefact_root": execution.artefact_root,
            "report_path": execution.report_path,
        },
    )
    plots = f"{execution.artefact_root}/plots" if execution.artefact_root else None
    data = {
        "config_path": execution.config_path,
        "success": True,
        "experiment_name": execution.experiment_name,
        "artefact_root": execution.artefact_root,
        "report_path": execution.report_path,
        "plots_dir": plots,
    }
    display = (
        f"Executed one config -> experiment '{execution.experiment_name}'. "
        f"Artefacts: {execution.artefact_root}. (single run; no loop, no retry)"
    )
    return _envelope(True, "execution_completed", display, data, "review_post_run_result")


def review_post_run_result(
    experiment_name: str,
    session_id: str | None = None,
    provider: str = "stub",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run an LLM review of the artefacts produced by a freshly executed config.

    Advisory only. Records POST_RUN_REVIEW_GENERATED when a session_id is given.
    """
    return _review_experiment(
        experiment_name,
        session_id,
        provider,
        model,
        base_url,
        SessionEventType.POST_RUN_REVIEW_GENERATED,
        "post_run_review_generated",
        "Post-run review",
        "get_session_summary",
    )


# ---------------------------------------------------------------------------
# Tool registration (explicit, safety-aware descriptions)
# ---------------------------------------------------------------------------

# Provider guidance for the LLM-backed tools. Shown to the model so it asks the
# user for the right LM Studio parameters and never requests a real OpenAI key.
_LLM_HINT = (
    " To use a local LM Studio model pass provider='openai', "
    "model='<exact LM Studio model id>', base_url='http://127.0.0.1:1234/v1' "
    "(when base_url is set, no real OpenAI API key is needed). "
    "provider='stub' gives a deterministic offline result with no model."
)

# Appended to every session-aware tool so the model passes the right identifier.
_SESSION_HINT = (
    " Pass session_id — the UUID returned by create_research_session — when "
    "available so this action is recorded in the session. Never use an "
    "experiment name, context_hash, or draft_id as session_id; if the UUID was "
    "lost, call get_latest_research_session to recover it."
)

_TOOLS: list[tuple[Any, str, str]] = [
    (get_zeto_operator_manual, "get_zeto_operator_manual",
     "Read-only: load the fixed Zeto operator manual (compact rules) at the "
     "start of a session. Executes nothing, calls no LLM, reads no files, "
     "mutates no state. Returns only the fixed rules — no arbitrary paths."),
    (list_experiments, "list_experiments",
     "List experiments that have persisted result artefacts. No side effects."),
    (create_research_session, "create_research_session",
     "Create a research session for an experiment with a stated research goal. "
     "Returns session_id (a UUID) — store it and pass it to all session-aware "
     "tools (review, proposal, draft, validate, approve, render, execute, "
     "post-run review, get_session_summary)."),
    (get_session_summary, "get_session_summary",
     "Return the current state of a research session. Requires session_id = the "
     "UUID returned by create_research_session — NOT an experiment name, "
     "context_hash, or draft_id. If the UUID was lost, call "
     "get_latest_research_session first."),
    (list_research_sessions, "list_research_sessions",
     "Read-only: list existing research session IDs (UUIDs). Use to recover a "
     "lost session_id. No side effects."),
    (get_latest_research_session, "get_latest_research_session",
     "Read-only: return the most recently updated research session (its UUID + "
     "a compact summary). Recovery helper for when the session_id was lost. The "
     "returned session_id is the UUID to pass to session-aware tools."),
    (check_research_workflow_state, "check_research_workflow_state",
     "Read-only preflight: report which artefacts exist for an experiment "
     "(baseline incl. metadata.json/metrics.json, review, proposal, latest "
     "draft + approval, rendered YAML, revised run, report/plots) and the "
     "recommended next step. Executes nothing. Call before major transitions "
     "to avoid out-of-order calls (e.g. validating before a draft exists)."),
    (build_context_summary, "build_context_summary",
     "Assemble deterministic structured context (failure modes, performance, "
     "validation) for an experiment. No LLM call, no experiment run."),
    (run_experiment_review, "run_experiment_review",
     "Run an advisory LLM review of an experiment's diagnostics. Interprets "
     "existing artefacts only; runs no experiment." + _LLM_HINT + _SESSION_HINT),
    (generate_iteration_proposal, "generate_iteration_proposal",
     "Generate an advisory next-step research proposal grounded in diagnostics. "
     "Prescribes no parameter values and runs no experiment." + _LLM_HINT + _SESSION_HINT),
    (generate_experiment_draft, "generate_experiment_draft",
     "Synthesise a schema-bounded config draft (typed deltas, not YAML). The "
     "draft is unapproved and must be validated and explicitly approved." + _LLM_HINT
     + " With provider='stub' it returns a deterministic draft "
       "(model.params.alpha -> 1.0)." + _SESSION_HINT),
    (validate_experiment_draft, "validate_experiment_draft",
     "Validate a draft against the authoritative config schema. No side effects."
     + _SESSION_HINT),
    (approve_experiment_draft, "approve_experiment_draft",
     "Explicitly approve a draft so it can be rendered. Only this tool approves; "
     "no other tool auto-approves. Ask the human before calling." + _SESSION_HINT),
    (render_draft_to_yaml, "render_draft_to_yaml",
     "Render an already-approved experiment draft to YAML. This does not execute "
     "the experiment. Execution requires a separate explicit tool call." + _SESSION_HINT),
    (execute_approved_config, "execute_approved_config",
     "Run exactly one approved YAML config through the quant research engine. "
     "Requires confirmation='RUN'. Does not optimise, loop, retry, trade, or "
     "register lineage automatically." + _SESSION_HINT),
    (review_post_run_result, "review_post_run_result",
     "Run an advisory LLM review of artefacts produced by a freshly executed "
     "config. Interprets existing artefacts only." + _LLM_HINT + _SESSION_HINT),
]

for _fn, _name, _desc in _TOOLS:
    mcp.add_tool(_fn, name=_name, description=_desc)


def main() -> None:
    """Run the Zeto MCP server over stdio (the transport LM Studio expects)."""
    mcp.run()


if __name__ == "__main__":
    main()
