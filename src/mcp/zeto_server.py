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


def _mfmt(value: Any) -> str:
    """Render a metric value for display; 'n/a' when missing. Never invents."""
    return "n/a" if value is None else str(value)


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


def _compact_multi_diff(changes: list[Any]) -> list[dict[str, Any]]:
    """Compact diff that handles feature add/remove alongside set changes."""
    result = []
    for c in changes:
        if c.section == "features":
            if c.field == "entries.add":
                feat = c.proposed_value
                result.append({
                    "field_path": "features.entries",
                    "operation": "add",
                    "value": (
                        {"name": feat.get("name"), "type": feat.get("type")}
                        if isinstance(feat, dict) else feat
                    ),
                })
            elif c.field == "entries.remove":
                result.append({
                    "field_path": "features.entries",
                    "operation": "remove",
                    "value": c.proposed_value,
                })
        else:
            result.append({
                "field_path": f"{c.section}.{c.field}",
                "operation": "set",
                "current_value": c.current_value,
                "proposed_value": c.proposed_value,
            })
    return result


def _diff_summary_lines(diff: list[dict[str, Any]]) -> list[str]:
    """One concise human-readable line per diff entry."""
    lines = []
    for d in diff:
        op = d.get("operation", "set")
        fp = d.get("field_path", d.get("field", "?"))
        if op == "set":
            lines.append(f"{fp}: {d.get('current_value')} -> {d.get('proposed_value')}")
        elif op == "add":
            v = d.get("value") or {}
            lines.append(
                f"features.entries ADD {v.get('name')!r} (type={v.get('type')!r})"
                if isinstance(v, dict) else f"features.entries ADD {v!r}"
            )
        elif op == "remove":
            lines.append(f"features.entries REMOVE {d.get('value')!r}")
        else:
            lines.append(f"{fp} [{op}]")
    return lines


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

    On any LLM failure or timeout this returns a clean ok=false envelope and
    stops — it never auto-retries the review (the user must ask again).
    """
    try:
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
    except Exception as exc:  # noqa: BLE001 — surface a clean envelope, never a traceback
        fail_stage = stage.replace("_generated", "_failed")
        return _envelope(
            False, fail_stage,
            f"{label} FAILED for '{experiment_name}': {type(exc).__name__}: {exc}. "
            "Stop and report the failure. Do not auto-retry unless the user asks.",
            {"experiment_name": experiment_name, "error": f"{type(exc).__name__}: {exc}"},
            "stop_and_report_to_user",
        )
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
    "If a session exists, always pass its session_id to approve/render/execute/"
    "post-run review; do not pass session_id=null while an active session exists.",
    "If session_id is lost, call get_latest_research_session before continuing.",
    "Never use experiment_name, context_hash, draft_id, or config_path as session_id.",
    "Never invent metrics; use tool outputs only.",
    "Never invent or hand-write draft configs; drafts come from generate_experiment_draft "
    "or generate_parameter_change_draft only.",
    "For a specific user-requested config change (e.g. 'set alpha to 2'), use "
    "generate_parameter_change_draft, not generate_experiment_draft.",
    "For Sharpe/OOS Sharpe/drawdown/hit-rate or 'did it improve' questions, use "
    "get_experiment_metrics or compare_experiment_metrics, never retrieve_research_memory "
    "or semantic_retrieve_research_memory.",
    "Never invent report contents or sample metrics; performance is authoritative only "
    "from experiment artefacts.",
    "If a tool returns ok=false, report the failure verbatim and stop.",
    "If validation fails, report the failure and stop; do not repeatedly call generate_experiment_draft.",
    "If validation fails because the proposed experiment name already exists, "
    "ask the user whether to use the next available suffix or clean existing local demo artefacts.",
    "Approval requires approval_confirmation='APPROVE'; validation success never authorises approval.",
    "Only a fresh user message that explicitly approves (prefer 'I approve the draft.') "
    "authorises approve_experiment_draft; never infer approval from 'start a research "
    "cycle' or a next_suggested_action.",
    "After approval_refused, stop and wait for the user; do not retry automatically.",
    "Do not render YAML unless the user approves the draft.",
    "Do not execute unless the user explicitly provides RUN (confirmation='RUN').",
    "Do not treat 'execute', 'yes', 'proceed', or 'continue' as RUN; only a fresh "
    "user message containing the literal token RUN authorises execution.",
    "Never infer RUN from a previous approval, a previous render, or a prior "
    "failed/refused execution attempt.",
    "After execution_refused, stop and wait for a new user message; do not retry "
    "execute_approved_config automatically.",
    "Do not run extra experiments, optimise automatically, or loop.",
    "A high-level request such as 'start a research cycle' does NOT authorise "
    "approval, rendering, or execution; it only authorises the read/advisory steps.",
    "Stop after validate_experiment_draft passes and ask the user for approval.",
    "Stop after render_draft_to_yaml and ask the user for RUN.",
    "Stop after the final get_session_summary; the research cycle is complete.",
    "Do not start a second iteration (new proposal, draft, or experiment) unless the user explicitly asks.",
    "Before review/proposal/draft, optionally call retrieve_research_memory or "
    "semantic_retrieve_research_memory for prior related evidence.",
    "Research memory is evidence-only; retrieved memory does not authorise execution or "
    "approval; semantic matches are suggestions, not proof.",
    "Quant metrics remain authoritative. If semantic retrieval fails, report and stop; "
    "do not invent prior evidence.",
    "For local Qwen-backed review/proposal/draft, use provider=openai, model=qwen2.5-7b-instruct, base_url=http://127.0.0.1:1234/v1.",
    "Governed sequence: create session -> check state -> execute baseline if "
    "needed -> build context -> review -> proposal -> draft -> validate -> "
    "approval -> render YAML -> RUN execution -> post-run review -> session summary.",
    "The quant engine remains authoritative; Qwen coordinates and interprets only.",
    "Config routing: 'what can we change?'->list_changeable_config_fields; "
    "'what features?'->list_available_features; 'which models?'->list_supported_models; "
    "'config overview?'->inspect_experiment_config. Never invent feature types or model names.",
    "For multiple changes or feature add/remove/replace, use generate_config_change_draft "
    "(changes is a JSON array). generate_parameter_change_draft handles a single set.",
    "If generate_config_change_draft fails, report the display verbatim and stop. "
    "Do not retry the same change. Use 'value' key in each change dict, never 'proposed_value'.",
    "For 'what did we learn from X vs Y?' or 'inspect evidence for X vs Y', use "
    "inspect_comparison_evidence (direct, authoritative). For 'have we tested X before?', "
    "use retrieve_research_memory(artefact_type=comparison_evidence). If no evidence "
    "exists, run compare_experiment_metrics first. Metrics remain authoritative.",
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
# Research memory tools (Phase 1 — metadata/keyword RAG)
#
# Evidence-only. These tools index compact summaries of existing artefacts and
# retrieve prior research evidence by deterministic keyword/metadata matching.
# They run no experiment, call no LLM, approve nothing, render nothing, and
# never authorise execution. Full artefacts stay on disk; only compact pointers
# (summaries, paths, hashes, tags, matched terms) are returned.
# ---------------------------------------------------------------------------


def get_research_memory_status() -> dict[str, Any]:
    """Read-only: report whether the research-memory index exists and its size."""
    status = _api.get_research_memory_status()
    if status["index_exists"]:
        display = (
            f"Research memory contains {status['item_count']} indexed item(s) "
            f"across {status['experiment_count']} experiment(s)."
        )
        next_action = "retrieve_research_memory"
    else:
        display = "Research memory index does not exist yet. Run index_research_memory."
        next_action = "index_research_memory"
    return _envelope(True, "research_memory_status", display, status, next_action)


def index_research_memory() -> dict[str, Any]:
    """Controlled write: build/refresh the memory index from known artefacts only.

    Runs no experiment, calls no LLM, approves/renders/executes nothing, and
    never mutates source artefacts. Reads only known Zeto artefact locations.
    """
    result = _api.index_research_memory()
    display = (
        f"Indexed {result['indexed_count']} research memory item(s) across "
        f"{result['experiment_count']} experiment(s)."
    )
    return _envelope(
        True, "research_memory_indexed", display, result, "retrieve_research_memory"
    )


def retrieve_research_memory(
    query: str | None = None,
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Read-only: retrieve prior research evidence by keyword/metadata match.

    Returns up to top_k compact items (summaries, paths, hashes, tags, matched
    terms) — never full artefact contents. Evidence only: a result authorises no
    execution, approval, or rendering; quant metrics remain authoritative.
    """
    items = _api.retrieve_research_memory(
        query=query,
        experiment_name=experiment_name,
        failure_modes=failure_modes,
        artefact_type=artefact_type,
        top_k=top_k,
    )
    focus = ", ".join(failure_modes) if failure_modes else (query or experiment_name or "recent")
    display = f"Retrieved {len(items)} related memory item(s) for {_short(focus, 80)}."
    data = {"item_count": len(items), "items": items}
    return _envelope(True, "research_memory_retrieved", display, data, "run_experiment_review")


# ---------------------------------------------------------------------------
# Semantic research memory tools (Phase 2 — local embedding retrieval)
#
# Evidence-only. These layer a local embedding index over the Phase 1 records
# and retrieve by cosine similarity. They embed compact summaries via a local
# embeddings endpoint only (never a chat/completion LLM), run no experiment,
# approve nothing, render nothing, and never authorise execution. Full artefacts
# stay on disk; only compact pointers (scores, summaries, paths, hashes, tags)
# are returned.
# ---------------------------------------------------------------------------


def get_semantic_research_memory_status() -> dict[str, Any]:
    """Read-only: report whether the semantic memory index exists and its size."""
    status = _api.get_semantic_research_memory_status()
    if status["index_exists"]:
        display = (
            f"Semantic research memory contains {status['item_count']} embedded "
            f"item(s) using {status['embedding_model']}."
        )
        next_action = "semantic_retrieve_research_memory"
    else:
        display = (
            "Semantic research memory index does not exist yet. Run "
            "index_semantic_research_memory."
        )
        next_action = "index_semantic_research_memory"
    return _envelope(True, "semantic_memory_status", display, status, next_action)


def index_semantic_research_memory(
    provider: str = "openai",
    model: str = "text-embedding-nomic-embed-text-v1.5",
    base_url: str = "http://127.0.0.1:1234/v1",
) -> dict[str, Any]:
    """Controlled write: embed Phase 1 memory records into a local semantic index.

    Embeds compact summaries via the local embeddings endpoint only — never a
    chat/completion LLM. Runs no experiment, approves/renders/executes nothing,
    mutates no source artefacts, and reads no arbitrary paths. Requires a Phase 1
    index first; on embedding-endpoint failure it returns a clean ok=false
    envelope and stops (no auto-retry).
    """
    result = _api.index_semantic_research_memory(
        provider=provider, model=model, base_url=base_url
    )
    status = result.get("status")
    if status == "no_phase1_index":
        return _envelope(
            False, "semantic_memory_index_blocked",
            "No Phase 1 research memory index found. Run index_research_memory first.",
            {"error": "Phase 1 memory index not found", **result},
            "index_research_memory",
        )
    if status == "embedding_failed":
        return _envelope(
            False, "semantic_memory_index_failed",
            "Embedding endpoint failed. Report the failure and stop; do not "
            f"auto-retry. Error: {_short(str(result.get('error')), 160)}",
            result,
            "ask_user_to_retry_semantic_index",
        )
    display = (
        f"Embedded {result['embedded_count']} research memory item(s) using "
        f"{result['embedding_model']}."
    )
    return _envelope(
        True, "semantic_memory_indexed", display, result, "semantic_retrieve_research_memory"
    )


def semantic_retrieve_research_memory(
    query: str,
    top_k: int = 5,
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    tags: list[str] | None = None,
    provider: str = "openai",
    model: str = "text-embedding-nomic-embed-text-v1.5",
    base_url: str = "http://127.0.0.1:1234/v1",
) -> dict[str, Any]:
    """Read-only: retrieve prior research evidence by local semantic similarity.

    Embeds the query with the same local embedding model and ranks the semantic
    index by cosine similarity (with optional metadata filters). Returns up to
    top_k compact items (scores, summaries, paths, hashes, tags, failure modes)
    — never full artefact bodies. Evidence only: authorises no execution,
    approval, or rendering; quant metrics remain authoritative. On any failure it
    returns a clean ok=false envelope and stops — it never invents evidence.
    """
    result = _api.semantic_retrieve_research_memory(
        query=query,
        top_k=top_k,
        experiment_name=experiment_name,
        failure_modes=failure_modes,
        artefact_type=artefact_type,
        tags=tags,
        provider=provider,
        model=model,
        base_url=base_url,
    )
    status = result.get("status")
    if status == "no_phase1_index":
        return _envelope(
            False, "semantic_memory_unavailable",
            "No Phase 1 research memory index found. Run index_research_memory first.",
            {"error": "Phase 1 memory index not found", "query": query, "items": []},
            "index_research_memory",
        )
    if status == "no_semantic_index":
        return _envelope(
            False, "semantic_memory_unavailable",
            "No semantic memory index found. Run index_semantic_research_memory first.",
            {"error": "Semantic memory index not found", "query": query, "items": []},
            "index_semantic_research_memory",
        )
    if status == "embedding_failed":
        return _envelope(
            False, "semantic_memory_retrieval_failed",
            "Embedding endpoint failed. Report the failure and stop; do not "
            f"auto-retry or invent evidence. Error: {_short(str(result.get('error')), 160)}",
            {"error": result.get("error"), "query": query, "items": []},
            "ask_user_to_retry_semantic_retrieval",
        )
    items = result.get("items", [])
    focus = ", ".join(failure_modes) if failure_modes else _short(query, 80)
    display = f"Retrieved {len(items)} semantically related memory item(s) for {focus}."
    data = {"query": query, "item_count": len(items), "items": items}
    return _envelope(
        True, "semantic_memory_retrieved", display, data, "run_experiment_review"
    )


# ---------------------------------------------------------------------------
# Explicit parameter-change draft (deterministic, no LLM)
#
# Use this when the user requests a specific config change ("set alpha to 2").
# It applies exactly that change, reads the real current value from the base
# config, and persists an unapproved draft. It never approves, renders, executes,
# or calls an LLM.
# ---------------------------------------------------------------------------


def generate_parameter_change_draft(
    experiment_name: str,
    field_path: str,
    proposed_value: Any,
    session_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Create a draft from one explicit, user-requested config change.

    Use this (NOT generate_experiment_draft) when the user asks for a specific
    change such as 'set model.params.alpha to 2'. Deterministic and LLM-free:
    applies exactly field_path -> proposed_value, reads the real current value
    from the base config, assigns a unique proposed name, validates against the
    schema, and persists an unapproved draft. Never approves, renders, executes,
    or retries. Invalid field paths and schema-incompatible values are refused.
    """
    result = _api.generate_parameter_change_draft(
        experiment_name=experiment_name,
        field_path=field_path,
        proposed_value=proposed_value,
        reason=reason,
    )
    status = result.get("status")
    if status != "ok":
        errors = [_short(e, 160) for e in result.get("errors", [])][:3]
        display = (
            f"Parameter-change draft refused ({status}): " + ("; ".join(errors) or "see errors")
            + ". Do not invent a fallback field or value; ask the user to correct the request."
        )
        return _envelope(
            False, "parameter_change_draft_failed", display,
            {"experiment_name": experiment_name, "field_path": field_path,
             "status": status, "errors": errors},
            "ask_user_to_correct_parameter_change",
        )

    draft = result["draft"]
    _record_event(
        session_id,
        SessionEventType.DRAFT_GENERATED,
        experiment_name,
        {"draft_id": draft.draft_id, "draft_hash": draft.draft_hash,
         "proposed_name": draft.proposed_name},
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
        f"Draft {draft.draft_id} '{draft.proposed_name}' created (approved="
        f"{draft.approved}). Proposed config diff: " + ("; ".join(diff_lines) or "(none)")
    )
    return _envelope(
        True, "parameter_change_draft_generated", display, data, "validate_experiment_draft"
    )


# ---------------------------------------------------------------------------
# Authoritative experiment metrics (read from artefacts, NOT from RAG memory)
#
# Use these to answer performance questions (Sharpe, OOS Sharpe, drawdown, hit
# rate, "did it improve?"). They read real on-disk diagnostics — never research
# memory. Compact output; full reports stay on disk.
# ---------------------------------------------------------------------------


def get_experiment_metrics(experiment_name: str) -> dict[str, Any]:
    """Read-only: authoritative metrics for one experiment from its artefacts.

    Use this (NOT retrieve_research_memory / semantic_retrieve_research_memory)
    to answer Sharpe / OOS Sharpe / drawdown / hit-rate questions. Returns compact
    metrics from real diagnostics with a missing_metrics list — never invented
    values. No LLM, no execution.
    """
    result = _api.get_experiment_metrics(experiment_name)
    if result.get("status") == "not_found":
        return _envelope(
            False, "experiment_metrics_unavailable",
            f"No metrics artefacts found for '{experiment_name}'. Do not invent "
            "metrics; verify the experiment name or run it first.",
            {"experiment_name": experiment_name, "metrics": {}, "missing_metrics": []},
            "list_experiments",
        )
    m = result["metrics"]
    fm = result.get("failure_modes") or []
    display = (
        f"Metrics for {experiment_name}: Sharpe={_mfmt(m.get('sharpe_ratio'))}, "
        f"mean OOS Sharpe={_mfmt(m.get('mean_oos_sharpe'))}, "
        f"MaxDD={_mfmt(m.get('max_drawdown_pct'))}. "
        f"Failure modes: {', '.join(fm) or 'none'}."
    )
    return _envelope(
        True, "experiment_metrics_loaded", display, result, "compare_experiment_metrics"
    )


def compare_experiment_metrics(
    base_experiment_name: str,
    candidate_experiment_name: str,
    session_id: str | None = None,
    research_question: str | None = None,
    tested_change: str | None = None,
) -> dict[str, Any]:
    """Read-only: compare two experiments using authoritative artefact metrics.

    Use this to answer 'did performance improve?' or 'what did we learn?'.
    Computes Sharpe / mean OOS Sharpe / max-drawdown deltas from real diagnostics
    — never RAG memory.  Persists a compact comparison_evidence record so future
    memory queries can retrieve before/after conclusions.  No LLM, no execution.

    Optional context fields enrich the evidence record:
      research_question: the question this comparison answers
      tested_change: what was changed (e.g. "added risk_adjusted_momentum_20")
    """
    result = _api.compare_experiment_metrics(
        base_experiment_name,
        candidate_experiment_name,
        session_id=session_id,
        research_question=research_question,
        tested_change=tested_change,
    )
    if result.get("status") == "not_found":
        missing = result.get("missing_experiments", [])
        return _envelope(
            False, "experiment_metrics_unavailable",
            f"Missing metrics artefacts for: {', '.join(missing) or 'unknown'}. "
            "Do not invent metrics.",
            result,
            "list_experiments",
        )
    data = dict(result)
    data["evidence_path"] = result.get("evidence_path")
    return _envelope(
        True, "experiment_metrics_compared", result["conclusion"], data,
        "get_session_summary",
    )


def inspect_comparison_evidence(
    base_experiment_name: str,
    candidate_experiment_name: str,
) -> dict[str, Any]:
    """Read-only: compact summary of a specific comparison evidence record.

    Directly reads the persisted comparison_evidence.json for the given
    base/candidate pair — no LLM, no RAG, no execution.  Returns what was
    tested, metric deltas, failure modes, and the research conclusion.  If no
    record exists, suggests running compare_experiment_metrics first.
    """
    result = _api.inspect_comparison_evidence(
        base_experiment_name, candidate_experiment_name
    )
    if result.get("status") == "not_found":
        return _envelope(
            False, "comparison_evidence_not_found",
            f"No comparison evidence for {base_experiment_name!r} vs "
            f"{candidate_experiment_name!r}. "
            "Run compare_experiment_metrics first to create the record.",
            result,
            "compare_experiment_metrics",
        )

    deltas = result.get("metric_deltas") or {}
    tested = result.get("tested_change") or ""
    base_modes = set(result.get("failure_modes_base") or [])
    cand_modes = result.get("failure_modes_candidate") or []

    delta_parts: list[str] = []
    d_sharpe = deltas.get("delta_sharpe")
    d_oos = deltas.get("delta_mean_oos_sharpe")
    d_dd = deltas.get("delta_max_drawdown_pct")
    if d_sharpe is not None:
        delta_parts.append(f"Sharpe {d_sharpe:+.3f}")
    if d_oos is not None:
        delta_parts.append(f"OOS Sharpe {d_oos:+.3f}")
    if d_dd is not None:
        dd_str = f"MaxDD {d_dd:+.2f}pp" if d_dd != 0.0 else "MaxDD unchanged"
        delta_parts.append(dd_str)

    if cand_modes:
        modes_str = (
            "failure modes persisted"
            if set(cand_modes) & base_modes
            else "new failure modes detected"
        )
    else:
        modes_str = "no failure modes"

    parts = [f"Comparison {base_experiment_name} -> {candidate_experiment_name}"]
    if tested:
        parts.append(f"tested {tested}")
    parts.append(", ".join(delta_parts) if delta_parts else "no numeric deltas")
    parts.append(modes_str)
    display = "; ".join(parts) + "."

    return _envelope(True, "comparison_evidence_inspected", display, result, None)


# ---------------------------------------------------------------------------
# Config introspection tools (read-only; no LLM, no execution)
# ---------------------------------------------------------------------------


def inspect_experiment_config(experiment_name: str) -> dict[str, Any]:
    """Read-only: compact summary of the current YAML config for an experiment.

    Returns model type/params, feature count/names, validation and execution
    settings, universe, and the list of changeable field paths.  Does NOT return
    the raw YAML — the payload is always compact.  Executes nothing, calls no LLM,
    reads no arbitrary paths.
    """
    result = _api.inspect_experiment_config(experiment_name)
    if result.get("status") == "config_not_found":
        return _envelope(
            False, "config_not_found",
            f"No config found for {experiment_name!r}. "
            + (result.get("errors") or [""])[0],
            {"experiment_name": experiment_name, "errors": result.get("errors", [])},
            "list_experiments",
        )
    model_type = (result.get("model") or {}).get("type", "unknown")
    feat_count = (result.get("features") or {}).get("count", 0)
    val_type = (result.get("validation") or {}).get("type", "unknown")
    display = (
        f"Config for {experiment_name!r}: model={model_type}, "
        f"{feat_count} feature(s), validation={val_type}."
    )
    return _envelope(
        True, "experiment_config_inspected", display,
        result,
        "list_changeable_config_fields",
    )


def list_changeable_config_fields(
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Read-only: list the controlled config-change surface — paths, operations, types.

    Returns the authoritative set of field paths that generate_config_change_draft
    will accept, with their operations (set / add / remove / replace) and current
    values when experiment_name is given.  Use this before building a draft so you
    know exactly what can and cannot be changed.
    """
    result = _api.list_changeable_config_fields(experiment_name)
    n = len(result.get("fields", []))
    entry_count = result.get("feature_entry_count")
    feat_info = (
        f"; current feature entries: {entry_count}" if entry_count is not None else ""
    )
    display = (
        f"{n} changeable field(s) across model, features, labels, signal, "
        f"validation, execution, portfolio_construction{feat_info}."
    )
    return _envelope(
        True, "changeable_config_fields_listed", display, result,
        "generate_config_change_draft",
    )


def list_available_features(
    experiment_name: str | None = None,
    family: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Read-only: valid feature types from the authoritative schema.

    Returns schema-validated feature types only — never invents names.  When
    experiment_name is given, shows which types are currently used.  Optional
    family/query filters narrow the list.  Use this to find valid types before
    calling generate_config_change_draft with an add/remove/replace operation.
    """
    result = _api.list_available_features(
        experiment_name=experiment_name, family=family, query=query
    )
    total = result.get("total_types", 0)
    used_count = result.get("currently_used_count")
    used_str = f"; {used_count} currently used" if used_count is not None else ""
    display = f"{total} valid feature type(s) in schema{used_str}."
    return _envelope(
        True, "available_features_listed", display, result,
        "generate_config_change_draft",
    )


def list_supported_models(
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Read-only: model types supported by the quant engine / config schema.

    Returns schema-authoritative model names only — never invents models.
    Model switching IS supported (via generate_config_change_draft with
    field_path='model.type', operation='set').  When experiment_name is given,
    highlights the current model type.
    """
    result = _api.list_supported_models(experiment_name=experiment_name)
    current = result.get("current_model") or "unknown"
    n = len(result.get("supported_models", []))
    switching = result.get("model_switching_supported", False)
    display = (
        f"Current model: {current}. {n} supported model type(s). "
        f"Model switching: {'supported' if switching else 'not supported'}."
    )
    return _envelope(
        True, "supported_models_listed", display, result,
        "generate_config_change_draft" if switching else None,
    )


def generate_config_change_draft(
    experiment_name: str,
    changes: str,
    session_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Deterministic multi-change config draft (NO LLM).

    ``changes`` is a JSON array of change dicts.  Supported per-change shapes:

      set:     {"field_path": "model.params.alpha", "operation": "set", "value": 2.0}
      add:     {"field_path": "features.entries", "operation": "add",
                "value": {"name": "...", "type": "...", "params": {...}}}
      remove:  {"field_path": "features.entries", "operation": "remove",
                "value": "feature_name"}
      replace: {"field_path": "features.entries", "operation": "replace",
                "old_value": "old_name", "value": {"name": "...", "type": "...", ...}}

    All changes are validated before any draft is created.  Returns ok=false on
    invalid field paths, unsupported feature types, absent features, duplicate
    adds, or schema-incompatible values — with no fallback invented.  The draft is
    unapproved.  Never approves, renders, executes, or calls an LLM.
    """
    # Parse the JSON changes string — refuse malformed input rather than guessing.
    try:
        parsed_changes: list[dict[str, Any]] = (
            changes if isinstance(changes, list) else __import__("json").loads(changes)
        )
    except Exception as exc:  # noqa: BLE001
        return _envelope(
            False, "config_change_draft_failed",
            f"'changes' must be a JSON array of change objects: {exc}",
            {"error": str(exc)},
            "list_changeable_config_fields",
        )

    result = _api.generate_config_change_draft(
        experiment_name, parsed_changes, reason=reason
    )
    status = result.get("status", "")

    if status != "ok":
        errors = result.get("errors", [])
        err_display = "; ".join(e for e in errors[:3]) if errors else "see errors"
        return _envelope(
            False, "config_change_draft_failed",
            f"Draft failed for {experiment_name!r}: {err_display}",
            {"experiment_name": experiment_name, "status": status, "errors": errors},
            "stop_and_report_to_user",
        )

    draft = result["draft"]
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
    diff = _compact_multi_diff(draft.changes)
    diff_lines = _diff_summary_lines(diff)
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft.draft_id,
        "proposed_name": draft.proposed_name,
        "approved": draft.approved,
        "diff": diff,
        "draft_path": _api.draft_artifact_path(experiment_name, draft.draft_id),
    }
    display = (
        f"Draft {draft.draft_id} '{draft.proposed_name}' "
        f"({len(diff)} change(s); approved={draft.approved}).\n  "
        + ("\n  ".join(diff_lines) or "(no changes)")
    )
    return _envelope(True, "config_change_draft_generated", display, data,
                     "validate_experiment_draft")


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
        f"approved config: {summary['approved_config_path'] or 'none'}. "
        "Research cycle complete — stop. Do not start another proposal, draft, or "
        "experiment unless the user explicitly asks."
    )
    return _envelope(True, "session_summary", display, data, "stop_cycle_complete")


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
    # A duplicate proposed name is the one recoverable failure — it has a clean
    # resolution (a new suffix or cleaning demo artefacts).  Every other failure
    # is non-recoverable here and the model must stop, not regenerate.
    duplicate_name = (not result.is_valid) and any(
        "already exists" in e for e in result.errors
    )
    data = {
        "experiment_name": experiment_name,
        "draft_id": draft_id,
        "is_valid": result.is_valid,
        "rendering_blocked": not result.is_valid,
        "duplicate_proposed_name": duplicate_name,
        "recoverable": duplicate_name,
        "error_count": len(result.errors),
        "errors": errors,
    }
    if result.is_valid:
        # Governance boundary: validation never authorises approval. Stop here and
        # hand control back to the user — do NOT auto-advance to approval.
        display = (
            "Validation PASS — approval still required. Stop and ask the user "
            "before approving or rendering."
        )
        next_action = "ask_user_for_approval"
    elif duplicate_name:
        # Recoverable, but still requires the user to choose — never auto-loop.
        display = (
            "Validation FAILED — proposed experiment name already exists; "
            "rendering is BLOCKED. Stop and ask the user whether to use the next "
            "available name suffix or clean existing local demo artefacts. Do not "
            "regenerate repeatedly."
        )
        next_action = "ask_user_resolve_duplicate_name"
    else:
        # Non-recoverable: stop and report, do NOT regenerate the same draft.
        display = (
            "Validation FAILED — rendering is BLOCKED. Stop and ask the user how "
            "to proceed. Do not regenerate repeatedly. Errors: " + "; ".join(errors)
        )
        next_action = "stop_and_report_to_user"
    return _envelope(result.is_valid, "draft_validated", display, data, next_action)


def approve_experiment_draft(
    experiment_name: str,
    draft_id: str,
    approval_confirmation: str = "",
    session_id: str | None = None,
    render_requested: bool = False,
) -> dict[str, Any]:
    """Explicitly approve a draft so it can be rendered to YAML.

    Requires approval_confirmation='APPROVE'. Approval is performed only by this
    tool — no other tool auto-approves — and only when the latest user message
    explicitly approves the draft.  Records DRAFT_APPROVED when a session_id is
    given.

    Call this ONLY when the latest user instruction explicitly approves the draft.
    Never infer approval from 'start a research cycle', from validation success,
    or from a next_suggested_action; never approve automatically after validation.

    render_requested: set True ONLY when the user explicitly asked to approve AND
    render in the same message. It does not render anything here; it only controls
    whether the suggested next step is render_draft_to_yaml (when True) or
    ask_user_to_render_yaml (when False — the default, which stops for the user).
    """
    if approval_confirmation != "APPROVE":
        # Tool-enforced approval gate: refuse before loading/mutating any state.
        # A refusal must NOT loop back into this tool — stop and wait for the user.
        return _envelope(
            False, "approval_refused",
            "Approval refused — confirmation='APPROVE' is required. Stop. Ask the "
            "user to explicitly approve before calling approve_experiment_draft again.",
            {"approved": False, "error": "Approval requires approval_confirmation='APPROVE'."},
            "ask_user_for_approval",
        )
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
        "render_requested": render_requested,
    }
    if render_requested:
        display = (
            f"Draft {approved.draft_id} approved at {approved.approved_at}. "
            "User requested approve+render in the same message — rendering may proceed."
        )
        next_action = "render_draft_to_yaml"
    else:
        display = (
            f"Draft {approved.draft_id} approved at {approved.approved_at}. "
            "Stop and ask the user before rendering YAML."
        )
        next_action = "ask_user_to_render_yaml"
    return _envelope(True, "draft_approved", display, data, next_action)


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
        f"Rendered {config_path}. Execution has NOT occurred. Stop. Execute only "
        "after the user explicitly says RUN."
    )
    return _envelope(
        True, "yaml_rendered", display, data, "ask_user_for_execution_authorisation"
    )


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

    Call this ONLY when the latest user instruction explicitly contains the
    literal token RUN. Never infer RUN from a previous approval or render, from
    "execute"/"yes"/"proceed"/"continue", or from a prior failed/refused attempt.
    """
    if confirmation != "RUN":
        # Governance boundary: a refusal must NOT loop back into this tool. Stop
        # and wait for a fresh user message that explicitly contains RUN.
        return _envelope(
            False, "execution_refused",
            "Execution refused — confirmation='RUN' is required. Stop. Ask the "
            "user to explicitly type RUN before calling execute_approved_config again.",
            {"success": False, "error": "Execution requires confirmation='RUN'."},
            "ask_user_for_execution_authorisation",
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
    (get_research_memory_status, "get_research_memory_status",
     "Read-only: report whether the research-memory index exists and how many "
     "items / experiments it holds. Evidence layer only — runs nothing."),
    (index_research_memory, "index_research_memory",
     "Controlled write: build/refresh the research-memory index from known Zeto "
     "artefact locations only (metadata, metrics, reviews, proposals, drafts, "
     "reports, sessions). Runs no experiment, calls no LLM, approves/renders/"
     "executes nothing, mutates no source artefacts, reads no arbitrary paths."),
    (retrieve_research_memory, "retrieve_research_memory",
     "Read-only: retrieve prior research evidence by deterministic keyword / "
     "metadata match (query, experiment_name, failure_modes, artefact_type, "
     "top_k). Returns compact items (summaries, paths, hashes, tags, matched "
     "terms) — never full artefacts. Evidence only: authorises no execution, "
     "approval, or rendering; quant metrics remain authoritative."),
    (get_semantic_research_memory_status, "get_semantic_research_memory_status",
     "Read-only: report whether the semantic (embedding) memory index exists, "
     "how many items it holds, and the embedding model. Evidence layer only."),
    (index_semantic_research_memory, "index_semantic_research_memory",
     "Controlled write: embed Phase 1 memory records into a local semantic index "
     "via a local embeddings endpoint only (provider/model/base_url; default "
     "LM Studio nomic). Calls no chat/completion LLM, runs no experiment, "
     "approves/renders/executes nothing, mutates no source artefacts, reads no "
     "arbitrary paths. Requires a Phase 1 index first."),
    (semantic_retrieve_research_memory, "semantic_retrieve_research_memory",
     "Read-only: retrieve prior research evidence by local semantic similarity "
     "(query embedded with the same model; optional experiment_name / "
     "failure_modes / artefact_type / tags filters; top_k). Returns compact items "
     "with cosine scores (summaries, paths, hashes, tags) — never full artefacts. "
     "Evidence only: suggestions, not proof; authorises no execution, approval, "
     "or rendering. On failure it stops and never invents evidence."),
    (inspect_experiment_config, "inspect_experiment_config",
     "Read-only: compact summary of an experiment's current YAML config — model "
     "type/params, feature count/names, validation settings, universe, and the "
     "list of changeable field paths. Does NOT return raw YAML. Executes nothing, "
     "calls no LLM, reads no arbitrary paths."),
    (list_changeable_config_fields, "list_changeable_config_fields",
     "Read-only: authoritative list of config field paths that generate_config_change_draft "
     "accepts, with their operations (set / add / remove / replace), types, allowed "
     "values, and current values when experiment_name is given. Call this before "
     "building a draft to know what can and cannot be changed."),
    (list_available_features, "list_available_features",
     "Read-only: schema-valid feature types only — never invents names. Includes "
     "family, required_params, and whether each type is currently used when "
     "experiment_name is given. Optional family/query filters. Use this to find valid "
     "feature types before calling generate_config_change_draft with add/remove/replace."),
    (list_supported_models, "list_supported_models",
     "Read-only: model types supported by the quant engine schema — never invents "
     "models. Model switching IS supported via generate_config_change_draft with "
     "field_path='model.type', operation='set'. Highlights the current model when "
     "experiment_name is given."),
    (generate_config_change_draft, "generate_config_change_draft",
     "Use this for multi-change or feature add/remove/replace drafts. Deterministic, "
     "no LLM: accepts a JSON-array 'changes' with set/add/remove/replace operations, "
     "validates all feature types against the schema, refuses unknown features/models/"
     "paths, assigns a unique name, and persists an UNAPPROVED draft. Never approves, "
     "renders, or executes. generate_parameter_change_draft remains available for "
     "simple single-field set operations." + _SESSION_HINT),
    (generate_parameter_change_draft, "generate_parameter_change_draft",
     "Use this (NOT generate_experiment_draft) when the user asks for a specific "
     "config change such as 'set model.params.alpha to 2'. Deterministic, no LLM: "
     "applies exactly field_path -> proposed_value, reads the real current value "
     "from the base config, assigns a unique name, validates, and persists an "
     "UNAPPROVED draft. Refuses invalid field paths and schema-incompatible values "
     "(no fallback field invented). Never approves, renders, or executes." + _SESSION_HINT),
    (get_experiment_metrics, "get_experiment_metrics",
     "Read-only: authoritative metrics (Sharpe, OOS Sharpe, drawdown, hit rate, "
     "failure modes) for one experiment, read from real artefacts. Use this — NOT "
     "retrieve_research_memory / semantic_retrieve_research_memory — to answer "
     "performance questions. Missing metrics are reported, never invented."),
    (compare_experiment_metrics, "compare_experiment_metrics",
     "Read-only: compare two experiments (Sharpe / mean OOS Sharpe / max-drawdown "
     "deltas + failure modes) using real artefact metrics, not RAG memory. Use this "
     "to answer whether performance improved. Missing metrics are reported clearly."),
    (inspect_comparison_evidence, "inspect_comparison_evidence",
     "Read-only: inspect a specific comparison evidence record for a base/candidate "
     "pair. Returns tested_change, metric deltas, failure modes, and conclusion from "
     "the persisted record — no LLM, no RAG. Direct authoritative summary. If no "
     "record exists, run compare_experiment_metrics first to create it."),
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
     "Explicitly approve a draft so it can be rendered. Requires "
     "approval_confirmation='APPROVE'. Only this tool approves; no other tool "
     "auto-approves. Call it ONLY when the latest user message explicitly approves "
     "the draft (e.g. 'I approve the draft.'). Never infer approval from 'start a "
     "research cycle', from validation success, or from a next_suggested_action; "
     "never approve automatically after validation. Pass render_requested=true "
     "ONLY if the user explicitly asked to approve AND render in the same message; "
     "otherwise the suggested next step is to stop and ask the user before "
     "rendering." + _SESSION_HINT),
    (render_draft_to_yaml, "render_draft_to_yaml",
     "Render an already-approved experiment draft to YAML. This does not execute "
     "the experiment. After rendering, STOP and ask the user for RUN — rendering "
     "never authorises execution." + _SESSION_HINT),
    (execute_approved_config, "execute_approved_config",
     "Run exactly one approved YAML config through the quant research engine. "
     "Requires confirmation='RUN'. Call this ONLY when the latest user instruction "
     "explicitly contains the literal token RUN. Never infer RUN from a previous "
     "approval, a previous render, the word 'execute'/'yes'/'proceed'/'continue', "
     "or a failed/refused execution attempt; after a refusal, stop and wait for a "
     "new user message. Does not optimise, loop, retry, trade, or register lineage "
     "automatically." + _SESSION_HINT),
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
