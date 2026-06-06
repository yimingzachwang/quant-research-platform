"""Governed AI-assisted research orchestration demo.

Walks the full advisory research cycle end-to-end against persisted experiment
artefacts — without ever running an experiment, touching the quant engine, or
making an autonomous decision:

    Research Session
      -> Structured LLM Context
      -> LLM Review
      -> Iteration Proposal
      -> Experiment Draft
      -> Draft Validation
      -> Governance Check (render-before-approval is blocked)
      -> Human Approval
      -> YAML Rendering
      -> Session Summary

The demo communicates one invariant:

    AI proposes.  The platform validates.  The researcher approves.
    The quant engine remains authoritative.

Usage:
    # Stub mode (default): deterministic, no API key, no network call.
    python scripts/demo_ai_orchestration.py --provider stub

    # LM Studio (OpenAI-compatible local server):
    export OPENAI_API_KEY=lm-studio
    python scripts/demo_ai_orchestration.py \\
        --provider openai \\
        --model <exact-lm-studio-model-id> \\
        --base-url http://127.0.0.1:1234/v1

    # Different experiment:
    python scripts/demo_ai_orchestration.py --experiment canonical_ml_showcase

The demo ends at YAML generation.  It NEVER runs the rendered config — the
researcher inspects and runs it manually later.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.api.research_api import (
    approve_experiment_draft,
    build_llm_context,
    create_research_session,
    execute_and_review_approved_config,
    generate_experiment_draft,
    generate_iteration_proposal,
    record_session_event,
    render_draft_to_yaml,
    run_llm_review,
    summarize_research_session,
    update_research_session_status,
    validate_experiment_draft,
)
from src.orchestration.llm.llm_interface import LLMResponse
from src.orchestration.llm.review_engine import _compute_context_hash
from src.orchestration.llm.review_schema import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    PROVIDER_STUB,
)
from src.orchestration.session.session_schema import SessionEventType, SessionStatus
from src.orchestration.utils.filesystem import experiment_config_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT = "canonical_ml_showcase"
DEFAULT_GOAL = "Investigate whether the model can improve validation robustness."
TOTAL_STAGES = 10
EXECUTION_TOTAL_STAGES = 13

# Mutable so _stage() can show the right denominator once we know whether the
# optional execution phase will run.  Set at the top of run_demo().
_TOTAL = TOTAL_STAGES

# Required artefacts (relative to results/experiments/<name>/) for the demo to run.
REQUIRED_ARTEFACTS = (
    "metadata.json",
    "metrics.json",
)
# Recommended artefacts — absence only weakens the review/proposal, not fatal.
RECOMMENDED_ARTEFACTS = (
    "diagnostics/ml_diagnostics.json",
    "diagnostics/split_metrics.json",
    "research/feature_summary.json",
    "plots/plot_index.json",
)

_RESULTS_ROOT = PROJECT_ROOT / "results" / "experiments"


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


def _layer(title: str, caption: str) -> None:
    """Print a five-layer architecture banner with its one-line role caption."""
    print(f"\n=== {title} ===")
    print(f"    {caption}")


def _stage(n: int, message: str) -> None:
    print(f"\n[{n}/{_TOTAL}] {message}")


def _line(label: str, value: object) -> None:
    print(f"    {label}: {value}")


# Layer role captions (suggested wording from the architecture spec).
_LAYER_PLANNING = (
    "LAYER 1 · AI PLANNING — a research request is routed into a supported "
    "workflow and translated into a governed configuration draft."
)
_LAYER_GOVERNANCE = (
    "LAYER 2 · HUMAN GOVERNANCE — the draft cannot become YAML until explicit "
    "approval is recorded."
)
_LAYER_ENGINE = (
    "LAYER 3 · QUANT RESEARCH ENGINE — execution is deliberately outside the AI "
    "loop. The researcher may manually run the rendered YAML after review."
)
_LAYER_CONTEXT = (
    "LAYER 4 · AI CONTEXT ENGINEERING — research artefacts are assembled into "
    "structured context before any LLM review."
)
_LAYER_MEMORY = (
    "LAYER 5 · AI REASONING & RESEARCH MEMORY — review, proposal, draft, "
    "approval, and rendered YAML are recorded as a research session."
)


# ---------------------------------------------------------------------------
# Argument parsing (importable + testable)
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_ai_orchestration",
        description="Governed AI-assisted research orchestration demo (advisory only).",
    )
    parser.add_argument(
        "--provider",
        default=PROVIDER_STUB,
        choices=[PROVIDER_STUB, PROVIDER_OPENAI, PROVIDER_ANTHROPIC],
        help="LLM provider. Default 'stub' (deterministic, no network, no key).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID. For LM Studio, the exact model id shown in the server.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:1234/v1 for LM Studio.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment name. Default '{DEFAULT_EXPERIMENT}'.",
    )
    parser.add_argument(
        "--goal",
        default=DEFAULT_GOAL,
        help="Research goal recorded on the session.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the final YAML in memory without writing it to disk.",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help=(
            "Also write the console transcript to "
            "results/demo/ai_orchestration_demo_transcript.txt (for docs/website)."
        ),
    )
    parser.add_argument(
        "--execute-approved",
        dest="execute_approved",
        action="store_true",
        help=(
            "Optionally run the approved YAML through the quant engine AFTER "
            "rendering. Requires explicit authorisation (typed RUN, or --yes)."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Skip the interactive RUN confirmation. Has NO effect unless "
            "--execute-approved is also passed."
        ),
    )
    parser.add_argument(
        "--execution-preset",
        dest="execution_preset",
        default="canonical",
        help="Report preset for the execution run (default: canonical).",
    )
    parser.add_argument(
        "--no-report",
        dest="no_report",
        action="store_true",
        help="Run the approved config without generating a report.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Artefact preflight
# ---------------------------------------------------------------------------


def check_artefacts(
    experiment: str,
    results_root: Path | None = None,
    configs_base: Path | None = None,
) -> list[str]:
    """Return a list of missing REQUIRED artefacts (empty list == ready).

    The experiment config YAML is also required (draft synthesis needs a
    version-2 ML config).  Recommended-but-optional artefacts are not included
    in the returned list.
    """
    root = (results_root or _RESULTS_ROOT) / experiment
    missing: list[str] = []
    for rel in REQUIRED_ARTEFACTS:
        if not (root / rel).exists():
            missing.append(f"results/experiments/{experiment}/{rel}")

    config_path = experiment_config_path(experiment, configs_base)
    if not Path(config_path).exists():
        missing.append(f"configs/experiments/{experiment}.yaml")
    return missing


def _print_missing_artefacts(experiment: str, missing: list[str]) -> None:
    print("\nMissing required artefacts — cannot run the demo:")
    for m in missing:
        print(f"  - {m}")
    print(
        "\nGenerate them first (this is a separate, researcher-initiated step "
        "and is NOT run automatically by this demo):\n"
    )
    print(
        f"  python scripts/run_from_config.py "
        f"configs/experiments/{experiment}.yaml --report --preset canonical\n"
    )


# ---------------------------------------------------------------------------
# Stub determinism shim
# ---------------------------------------------------------------------------


def stub_draft_json(experiment: str) -> str:
    """A deterministic, schema-valid draft response for the stub provider.

    The stub LLM returns a fixed placeholder string that is not valid JSON, so
    the real draft-synthesis path cannot parse it.  For the stub provider only,
    this supplies a minimal schema-conforming change set (stronger L2
    regularisation — a plausible move toward validation robustness), mirroring
    how the test suite injects a canned response.  Real providers are untouched.
    """
    return json.dumps(
        {
            "proposed_name": f"{experiment}_v2",
            "changes": [
                {
                    "section": "model",
                    "field": "params.alpha",
                    "proposed_value": 1.0,
                    "rationale": (
                        "Increase L2 regularisation to stabilise coefficients and "
                        "improve out-of-sample validation consistency."
                    ),
                }
            ],
        }
    )


@contextmanager
def _stub_draft_shim(provider: str, experiment: str):
    """Patch the draft generator's LLM call to emit valid JSON for stub mode.

    Only active when provider == 'stub'.  For any real provider this is a no-op
    and the genuine model response is parsed.
    """
    if provider != PROVIDER_STUB:
        yield
        return

    from unittest.mock import patch

    def _fake_call_llm(*_args, **_kwargs) -> LLMResponse:
        return LLMResponse(
            text=stub_draft_json(experiment),
            model="stub",
            provider=PROVIDER_STUB,
            usage={},
        )

    with patch(
        "src.orchestration.config_generation.draft_generator.call_llm",
        _fake_call_llm,
    ):
        yield


# ---------------------------------------------------------------------------
# Governed execution gate (importable + testable)
# ---------------------------------------------------------------------------


def execution_authorised(args: argparse.Namespace, input_fn=input) -> bool:
    """Return True only when execution is explicitly authorised.

    Governance rules enforced here:
      - execution requires --execute-approved (––yes alone never authorises);
      - without --yes, the researcher must type exactly RUN to proceed.
    """
    if not getattr(args, "execute_approved", False):
        return False
    if getattr(args, "yes", False):
        return True
    print("\nThis will run the approved YAML config through the quant research engine.")
    reply = input_fn("Type RUN to continue: ")
    return reply.strip() == "RUN"


def _run_execution_phase(
    session,
    experiment: str,
    config_path: object,
    args: argparse.Namespace,
    input_fn=input,
):
    """Optional, human-authorised execution + post-run review (stages 11–13).

    Returns (exit_code, session).  Never loops, never retries, runs exactly one
    config.  Reaching execution is gated on explicit authorisation.
    """
    _layer("HUMAN-CONTROLLED EXECUTION BOUNDARY", _LAYER_ENGINE)
    print("    The AI loop has ended at approved YAML.")
    print("    Execution requires explicit researcher authorisation.")

    # ---- Stage 11 — Authorisation gate -----------------------------------
    _stage(11, "Awaiting explicit execution authorisation...")
    if not execution_authorised(args, input_fn=input_fn):
        print("\nExecution cancelled. Demo remains complete at rendered YAML.")
        return 0, session, False

    session = record_session_event(
        session,
        event_type=SessionEventType.EXECUTION_REQUESTED,
        experiment_name=experiment,
        data={
            "config_path": str(config_path),
            "preset": args.execution_preset,
            "report": not args.no_report,
        },
    )

    # ---- Stage 12 — Run exactly one approved config ----------------------
    _stage(12, "Running approved YAML through quant research engine...")
    result = execute_and_review_approved_config(
        config_path,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        report=not args.no_report,
        preset=args.execution_preset,
    )
    execution = result.execution
    if not execution.success:
        print(f"    execution: FAILED — {execution.error}")
        print("    (no post-run review — execution did not produce artefacts)")
        return 1, session, False

    session = record_session_event(
        session,
        event_type=SessionEventType.EXECUTION_COMPLETED,
        experiment_name=execution.experiment_name or experiment,
        data={
            "config_path": execution.config_path,
            "experiment_name": execution.experiment_name,
            "artefact_root": execution.artefact_root,
            "report_path": execution.report_path,
        },
    )
    _line("execution", "SUCCESS")
    _line("new experiment", execution.experiment_name)
    _line("artefact root", execution.artefact_root)
    _line("report path", execution.report_path)

    # ---- Stage 13 — Review of the freshly generated artefacts ------------
    _stage(13, "Reviewing generated artefacts...")
    if result.review is not None:
        session = record_session_event(
            session,
            event_type=SessionEventType.POST_RUN_REVIEW_GENERATED,
            experiment_name=execution.experiment_name or experiment,
            data={
                "experiment_name": execution.experiment_name,
                "context_hash": result.context_hash,
                "provider": args.provider,
            },
        )
        _line("post-run context hash", result.context_hash)
        _line(
            "post-run review sections",
            ", ".join(result.review.sections.keys()) or "(none parsed)",
        )
    else:
        _line("post-run review", "skipped")
    return 0, session, True


# ---------------------------------------------------------------------------
# Demo orchestration
# ---------------------------------------------------------------------------


def run_demo(args: argparse.Namespace) -> int:
    experiment = args.experiment
    provider = args.provider

    global _TOTAL
    _TOTAL = EXECUTION_TOTAL_STAGES if args.execute_approved else TOTAL_STAGES

    print("=" * 70)
    print("AI Orchestration Demo — governed, advisory research cycle")
    print("=" * 70)
    _line("experiment", experiment)
    _line("provider", provider)
    if args.base_url:
        _line("base_url", args.base_url)
    if args.model:
        _line("model", args.model)
    print("\nFive-layer AI orchestration architecture:")
    print("    1. AI Planning Layer")
    print("    2. Human Governance Layer")
    print("    3. Quant Research Engine")
    print("    4. AI Context Engineering")
    print("    5. AI Reasoning & Research Memory")
    print(
        "\nDoctrine: AI proposes. The platform validates. The researcher approves. "
        "The quant engine remains authoritative."
    )

    # Preflight: required artefacts must exist.
    missing = check_artefacts(experiment)
    if missing:
        _print_missing_artefacts(experiment, missing)
        return 2

    # ---- Layer 5 (open) — Research session record ------------------------
    _layer("AI REASONING & RESEARCH MEMORY (session opened)", _LAYER_MEMORY)
    # ---- Stage 1 — Create research session -------------------------------
    _stage(1, "Creating research session...")
    session = create_research_session(
        root_experiment=experiment,
        research_goal=args.goal,
    )
    _line("session id", session.session_id)
    _line("research goal", session.research_goal)

    # ---- Layer 4 — AI Context Engineering --------------------------------
    _layer("AI CONTEXT ENGINEERING", _LAYER_CONTEXT)
    # ---- Stage 2 — Build structured LLM context (no LLM call) -------------
    _stage(2, "Building structured LLM context (deterministic, no LLM call)...")
    context = build_llm_context(experiment, persist=False)
    context_hash = _compute_context_hash(context)
    assert context.experiment_name == experiment, "context built for wrong experiment"
    _line("experiment", context.experiment_name)
    _line("context hash", context_hash)
    if context.failure_modes:
        print("    detected failure modes:")
        for fm in context.failure_modes:
            print(f"      - [{fm.get('severity', '?')}] {fm.get('name', '?')}")
    else:
        print("    detected failure modes: none")
    val = context.validation or {}
    if val:
        _line("mean OOS sharpe", val.get("mean_oos_sharpe"))
        _line("std OOS sharpe", val.get("std_oos_sharpe"))
        _line("n splits", val.get("n_splits"))

    # ---- Layer 1 + Layer 5 — AI Planning, grounded by AI Reasoning -------
    _layer("AI PLANNING LAYER  (grounded by AI Reasoning)", _LAYER_PLANNING)
    print(f"    {_LAYER_MEMORY}")
    # ---- Stage 3 — LLM review --------------------------------------------
    _stage(3, "Running LLM review...")
    review = run_llm_review(
        experiment,
        provider=provider,
        model=args.model,
        base_url=args.base_url,
        persist_context=False,
        persist_review=True,
    )
    assert review is not None, "review generation failed"
    session = record_session_event(
        session,
        event_type=SessionEventType.REVIEW_GENERATED,
        experiment_name=experiment,
        data={"provider": provider},
    )
    _line("context hash", context_hash)
    _line("review sections", ", ".join(review.sections.keys()) or "(none parsed)")
    _line("flags", ", ".join(review.flags) or "none")

    # ---- Stage 4 — Iteration proposal ------------------------------------
    _stage(4, "Generating iteration proposal...")
    proposal = generate_iteration_proposal(
        experiment,
        provider=provider,
        model=args.model,
        base_url=args.base_url,
        persist=True,
    )
    assert proposal is not None, "iteration proposal failed"
    session = record_session_event(
        session,
        event_type=SessionEventType.ITERATION_PROPOSAL_GENERATED,
        experiment_name=experiment,
        data={
            "context_hash": proposal.context_hash,
            "research_focus": proposal.research_focus,
        },
    )
    _line("research focus", proposal.research_focus or "(none)")
    if proposal.suggested_experiments:
        print("    suggested experiments:")
        for s in proposal.suggested_experiments:
            print(f"      - {s}")

    # ---- Stage 5 — Experiment draft (LLM proposes deltas, not YAML) -------
    _stage(5, "Generating experiment draft (deltas only)...")
    with _stub_draft_shim(provider, experiment):
        draft = generate_experiment_draft(
            experiment,
            provider=provider,
            model=args.model,
            base_url=args.base_url,
        )
    assert draft is not None, "draft generation failed"
    assert draft.approved is False, "draft must be unapproved on creation"
    session = record_session_event(
        session,
        event_type=SessionEventType.DRAFT_GENERATED,
        experiment_name=experiment,
        data={
            "draft_id": draft.draft_id,
            "draft_hash": draft.draft_hash,
            "proposed_name": draft.proposed_name,
        },
    )
    _line("draft id", draft.draft_id)
    _line("proposed config name", draft.proposed_name)
    _line("approved", draft.approved)
    print("    proposed changes:")
    for ch in draft.changes:
        print(f"      - {ch.section}.{ch.field}: {ch.current_value} -> {ch.proposed_value}")
        print(f"        rationale: {ch.rationale}")

    # ---- Layer 2 — Human Governance --------------------------------------
    _layer("HUMAN GOVERNANCE LAYER", _LAYER_GOVERNANCE)
    # ---- Stage 6 — Validate draft ----------------------------------------
    _stage(6, "Validating draft against the config schema...")
    result = validate_experiment_draft(draft)
    session = record_session_event(
        session,
        event_type=SessionEventType.DRAFT_VALIDATED,
        experiment_name=experiment,
        data={
            "draft_id": draft.draft_id,
            "is_valid": result.is_valid,
            "error_count": len(result.errors),
        },
    )
    _line("validation", "PASS" if result.is_valid else "FAIL")
    if result.errors:
        for err in result.errors:
            print(f"      - {err}")
    if not result.is_valid:
        print("\nDraft validation failed — stopping before approval/rendering.")
        return 1

    # ---- Stage 7 — Governance check: render must be blocked pre-approval --
    _stage(7, "Governance check: rendering YAML before approval must be blocked...")
    render_blocked = False
    try:
        render_draft_to_yaml(draft, dry_run=True)
    except ValueError:
        render_blocked = True
    if not render_blocked:
        print("Unapproved YAML render blocked: FAIL")
        print("\nGOVERNANCE VIOLATION: unapproved draft rendered to YAML. Aborting.")
        return 1
    print("Unapproved YAML render blocked: PASS")

    # ---- Stage 8 — Human approval ----------------------------------------
    _stage(8, "Approving draft (explicit human gate)...")
    approved = approve_experiment_draft(draft)
    assert approved.approved is True, "approval did not set approved=True"
    session = record_session_event(
        session,
        event_type=SessionEventType.DRAFT_APPROVED,
        experiment_name=experiment,
        data={"draft_id": approved.draft_id, "draft_hash": approved.draft_hash},
    )
    _line("approved", approved.approved)
    _line("approved at", approved.approved_at)

    # ---- Approved configuration — render YAML ----------------------------
    _layer("APPROVED CONFIGURATION", _LAYER_GOVERNANCE)
    # ---- Stage 9 — Render YAML (governance boundary) ---------------------
    _stage(9, "Rendering approved draft to YAML...")
    # Execution needs the file on disk, so --dry-run is ignored when the
    # researcher has asked to execute the approved config.
    render_dry = args.dry_run and not args.execute_approved
    if args.dry_run and args.execute_approved:
        print("    (note: --dry-run overridden — --execute-approved needs the YAML on disk)")
    yaml_str = render_draft_to_yaml(approved, dry_run=render_dry)
    config_path = experiment_config_path(approved.proposed_name)
    assert approved.draft_hash in yaml_str, "rendered YAML missing draft provenance"
    session = record_session_event(
        session,
        event_type=SessionEventType.YAML_RENDERED,
        experiment_name=experiment,
        data={"draft_id": approved.draft_id, "config_path": str(config_path)},
    )
    _line("rendered YAML path", config_path if not render_dry else f"{config_path} (dry-run, not written)")
    _line("draft hash (provenance)", approved.draft_hash)
    _line("source proposal hash", approved.source_proposal_hash)
    print("    (the rendered config is NOT executed by this demo)")

    # ---- Layer 3 — Quant Research Engine (manual execution boundary) -----
    _layer("QUANT RESEARCH ENGINE (manual execution boundary)", _LAYER_ENGINE)
    print("    The AI loop ends here. To run the approved config, the researcher")
    print("    manually executes (this demo never does):")
    print(
        f"      python scripts/run_from_config.py "
        f"configs/experiments/{approved.proposed_name}.yaml --report --preset canonical"
    )

    # ---- Layer 5 (close) — Research memory summary -----------------------
    _layer("AI REASONING & RESEARCH MEMORY (session summary)", _LAYER_MEMORY)
    # ---- Stage 10 — Session summary --------------------------------------
    _stage(10, "Session summary...")
    update_research_session_status(session, status=SessionStatus.COMPLETE)
    summary = summarize_research_session(session)
    _line("session id", summary["session_id"])
    _line("research goal", summary["research_goal"])
    _line("event count", summary["event_count"])
    _line("active experiment", summary["active_experiment"])
    _line("active draft", summary["active_draft"])
    _line("approved config path", summary["approved_config_path"])
    _line("experiments visited", ", ".join(summary["experiments_visited"]))

    # ---- Required governance assertions ----------------------------------
    assert summary["active_draft"] is None, "active_draft should be None after YAML render"
    assert summary["approved_config_path"] is not None, "approved config path must be set"
    recorded = {ev.event_type for ev in session.events}
    expected = {
        SessionEventType.REVIEW_GENERATED,
        SessionEventType.ITERATION_PROPOSAL_GENERATED,
        SessionEventType.DRAFT_GENERATED,
        SessionEventType.DRAFT_VALIDATED,
        SessionEventType.DRAFT_APPROVED,
        SessionEventType.YAML_RENDERED,
    }
    assert expected.issubset(recorded), f"session missing events: {expected - recorded}"

    # ---- Optional human-controlled execution -----------------------------
    if not args.execute_approved:
        print("\nExecution: not requested. Demo ends at rendered YAML.")
        print("\nAI orchestration demo complete.")
        return 0

    code, session, executed = _run_execution_phase(session, experiment, config_path, args)
    if code != 0:
        return code

    if not executed:
        print("\nAI orchestration demo complete.")
        return 0

    final = summarize_research_session(session)
    print("\nUpdated session after execution:")
    _line("event count", final["event_count"])
    _line("experiments visited", ", ".join(final["experiments_visited"]))

    print("\nAI orchestration demo complete (with execution).")
    return 0


class _Tee:
    """Minimal stdout fan-out: write to the real console and an in-memory buffer.

    Used only by --transcript.  This is not a logging framework — it just lets
    the same console output be captured for docs/website copy.
    """

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _run(args: argparse.Namespace) -> int:
    try:
        return run_demo(args)
    except Exception as exc:  # noqa: BLE001 — demo wants a clean, single-line failure
        print(f"\nDemo failed: {type(exc).__name__}: {exc}")
        if args.provider == PROVIDER_OPENAI and args.base_url:
            print(
                "\nIf using LM Studio, confirm the local server is running:\n"
                f"  curl {args.base_url.rstrip('/')}/models\n"
                "and that --model matches an exact model id from the server."
            )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.transcript:
        return _run(args)

    buffer = io.StringIO()
    tee = _Tee(sys.stdout, buffer)
    try:
        with contextlib.redirect_stdout(tee):
            code = _run(args)
    finally:
        path = PROJECT_ROOT / "results" / "demo" / "ai_orchestration_demo_transcript.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(buffer.getvalue(), encoding="utf-8")
        print(f"\nTranscript written to: {path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
