"""Top-level structured research API.

This is the single entry-point for the orchestration layer.  All calls from
external code (LLM front-ends, notebooks, CLI tools) should go through this
module.  It composes retrieval, context building, and LLM review into a
clean, typed interface.

The quantitative engine is NEVER called from here — only persisted artefacts
are read.  To run a new experiment, use src.experiments.orchestrator directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.api.artefact_api import (
    get_all_diagnostics,
    get_artefact,
    get_plots,
    list_experiment_artefacts,
)
from src.orchestration.api.comparison_api import (
    compare_experiments,
    diff_experiments,
    rank_experiments,
)
from src.orchestration.api.experiment_loader import (
    load_experiment_bundle,
)
from src.orchestration.api.schemas import (
    ArtefactMetadata,
    ComparativeReview,
    ExperimentLineage,
    ExperimentSummary,
    IterationProposal,
    LLMContext,
    LLMReviewOutput,
    PlotMetadata,
    ResearchEvolutionChain,
)
from src.orchestration.config_generation.draft_generator import generate_draft, load_draft
from src.orchestration.config_generation.draft_schema import (
    DraftValidationResult,
    ExperimentDraft,
)
from src.orchestration.config_generation.draft_validator import (
    approve_draft,
    validate_draft,
)
from src.orchestration.config_generation.yaml_renderer import render_to_yaml
from src.orchestration.context.context_builder import (
    build_and_persist_context,
    build_context,
)
from src.orchestration.evolution.evolution_builder import (
    build_evolution_chain,
    persist_evolution_chain,
    register_lineage,
)
from src.orchestration.llm.comparison_engine import run_comparative_review
from src.orchestration.llm.iteration_engine import run_iteration_proposal
from src.orchestration.llm.review_engine import run_review
from src.orchestration.llm.review_schema import PROVIDER_ANTHROPIC
from src.orchestration.registry.experiment_registry import (
    find_by_strategy,
    find_by_tag,
    get_summary,
    list_all,
    list_summaries,
    rank_by_sharpe,
)
from src.orchestration.session.session_manager import (
    create_session as _create_session,
)
from src.orchestration.session.session_manager import (
    load_session as _load_session,
)
from src.orchestration.session.session_manager import (
    record_event as _record_event,
)
from src.orchestration.session.session_manager import (
    summarize_session as _summarize_session,
)
from src.orchestration.session.session_manager import (
    update_session_status as _update_session_status,
)
from src.orchestration.session.session_schema import ResearchSession
from src.orchestration.utils.filesystem import list_session_ids

# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------


def list_all_experiments(base: Path | str | None = None) -> list[str]:
    """Return names of all experiments available on disk."""
    return list_all(base)


def find_experiments(
    tag: str | None = None,
    strategy_pattern: str | None = None,
    base: Path | str | None = None,
) -> list[str]:
    """Filter experiments by tag or strategy name pattern."""
    if tag:
        return find_by_tag(tag, base)
    if strategy_pattern:
        return find_by_strategy(strategy_pattern, base)
    return list_all(base)


def get_experiment_summary(
    experiment_name: str,
    base: Path | str | None = None,
) -> ExperimentSummary | None:
    return get_summary(experiment_name, base)


def list_experiment_summaries(base: Path | str | None = None) -> list[ExperimentSummary]:
    return list_summaries(base)


def rank_experiments_by_sharpe(
    base: Path | str | None = None,
    descending: bool = True,
) -> list[ExperimentSummary]:
    return rank_by_sharpe(base, descending)


# ---------------------------------------------------------------------------
# Artefact access
# ---------------------------------------------------------------------------


def load_experiment(
    experiment_name: str,
    base: Path | str | None = None,
    include_timeseries: bool = False,
) -> dict[str, Any]:
    """Load experiment metadata + metrics (+ optionally time-series)."""
    return load_experiment_bundle(experiment_name, base, include_timeseries)


def retrieve_artefact(
    experiment_name: str,
    key: str,
    base: Path | str | None = None,
) -> Any:
    """Load a single artefact by key."""
    return get_artefact(experiment_name, key, base)


def retrieve_all_diagnostics(
    experiment_name: str,
    base: Path | str | None = None,
) -> dict[str, Any]:
    return get_all_diagnostics(experiment_name, base)


def list_artefacts(
    experiment_name: str,
    base: Path | str | None = None,
) -> list[ArtefactMetadata]:
    return list_experiment_artefacts(experiment_name, base)


def get_experiment_plots(
    experiment_name: str,
    base: Path | str | None = None,
    primary_only: bool = False,
) -> list[PlotMetadata]:
    return get_plots(experiment_name, base, primary_only)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    experiment_names: list[str],
    base: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return a metric comparison table for the named experiments."""
    return compare_experiments(experiment_names, base)


def diff(
    name_a: str,
    name_b: str,
    base: Path | str | None = None,
) -> dict[str, Any]:
    return diff_experiments(name_a, name_b, base)


def rank(
    base: Path | str | None = None,
    metric: str = "sharpe_ratio",
    descending: bool = True,
) -> list[dict[str, Any]]:
    return rank_experiments(base, metric, descending)


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_llm_context(
    experiment_name: str,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
    persist: bool = True,
    llm_base: Path | str | None = None,
) -> LLMContext:
    """Build a structured LLMContext from persisted artefacts.

    If ``persist=True``, writes llm_context.json to results/llm_reviews/.
    """
    if persist:
        return build_and_persist_context(experiment_name, base, reports_base, llm_base)
    return build_context(experiment_name, base, reports_base)


# ---------------------------------------------------------------------------
# LLM review
# ---------------------------------------------------------------------------


def run_llm_review(
    experiment_name: str,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
    llm_base: Path | str | None = None,
    persist_context: bool = True,
    persist_review: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> LLMReviewOutput:
    """Build LLM context and run a review in one call.

    This is the highest-level entry point for AI-assisted experiment
    interpretation.  It:
      1. Loads all artefacts from disk.
      2. Builds a structured LLMContext.
      3. Renders the prompt template.
      4. Calls the selected LLM provider.
      5. Parses and persists the structured output.

    The quantitative engine is not involved — only persisted artefacts are read.
    """
    context = build_llm_context(
        experiment_name,
        base=base,
        reports_base=reports_base,
        persist=persist_context,
        llm_base=llm_base,
    )
    return run_review(
        context=context,
        provider=provider,
        model=model,
        persist=persist_review,
        llm_base=llm_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_llm_comparative_review(
    baseline_experiment: str,
    candidate_experiment: str,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    base: Path | str | None = None,
    comparisons_base: Path | str | None = None,
    persist: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    base_url: str | None = None,
) -> ComparativeReview:
    """Compare two experiments through diagnostics-aware semantic interpretation.

    Loads both experiment artefacts from disk, builds structured LLMContexts,
    assembles a comparative diagnostic payload with pre-computed deltas, renders
    the comparison prompt, calls the selected LLM provider, and returns a parsed
    ComparativeReview with full provenance metadata.

    Advisory only — interprets research evolution, does not rank experiments for
    deployment or recommend configuration changes.

    Flow:
        Baseline artefacts → semantic context
        Candidate artefacts → semantic context
        Comparative payload (pre-computed deltas)
        → Render prompt → LLM provider → Parse ComparativeReview → Persist
    """
    baseline_ctx = build_context(baseline_experiment, base)
    candidate_ctx = build_context(candidate_experiment, base)
    return run_comparative_review(
        baseline_ctx=baseline_ctx,
        candidate_ctx=candidate_ctx,
        provider=provider,
        model=model,
        persist=persist,
        comparisons_base=comparisons_base,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


def generate_iteration_proposal(
    experiment_name: str,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
    llm_base: Path | str | None = None,
    persist: bool = True,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    base_url: str | None = None,
) -> IterationProposal:
    """Generate a structured research iteration proposal from persisted artefacts.

    Loads all experiment artefacts from disk, builds a structured LLMContext,
    renders the iteration prompt, calls the selected LLM provider, and returns
    a parsed IterationProposal with full provenance metadata.

    The proposal is advisory only — the researcher remains the decision-maker
    for all experiment design and execution.

    Flow:
        Experiment artefacts → Context builder → Iteration prompt
        → LLM provider → Parsed IterationProposal → Persist artefacts
    """
    context = build_llm_context(
        experiment_name,
        base=base,
        reports_base=reports_base,
        persist=False,
        llm_base=llm_base,
    )
    return run_iteration_proposal(
        context=context,
        provider=provider,
        model=model,
        persist=persist,
        llm_base=llm_base,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Research evolution
# ---------------------------------------------------------------------------


def register_experiment_lineage(
    experiment_name: str,
    parent_experiment: str | None,
    iteration_reason: str | None = None,
    derived_from_iteration: bool = False,
    derived_from_comparison: bool = False,
    base: Path | str | None = None,
) -> ExperimentLineage:
    """Register lineage metadata for an experiment.

    Human-triggered only — does not start orchestration or modify experiments.
    Records why this experiment was created and which experiment it follows.
    Overwrites any existing lineage record for this experiment.
    """
    context_hash = ""
    try:
        ctx = build_context(experiment_name, base)
        from src.orchestration.llm.review_engine import _compute_context_hash
        context_hash = _compute_context_hash(ctx)
    except Exception:
        pass

    return register_lineage(
        experiment_name=experiment_name,
        parent_experiment=parent_experiment,
        iteration_reason=iteration_reason,
        derived_from_iteration=derived_from_iteration,
        derived_from_comparison=derived_from_comparison,
        context_hash=context_hash,
        experiments_base=base,
    )


# ---------------------------------------------------------------------------
# Config synthesis (Phase 3)
# ---------------------------------------------------------------------------


def load_experiment_draft(
    experiment_name: str,
    draft_id: str,
    llm_base: Path | str | None = None,
) -> ExperimentDraft | None:
    """Load a persisted ExperimentDraft by experiment name and draft ID.

    Returns None if the draft file does not exist.  Never raises.
    """
    return load_draft(experiment_name, draft_id, llm_base)


def generate_experiment_draft(
    experiment_name: str,
    provider: str = PROVIDER_ANTHROPIC,
    model: str | None = None,
    proposal_hash: str | None = None,
    base: Path | str | None = None,
    llm_base: Path | str | None = None,
    configs_base: Path | str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> ExperimentDraft:
    """Generate a typed ExperimentDraft from the most recent IterationProposal.

    Requires an IterationProposal already persisted for the experiment
    (run generate_iteration_proposal() first).  Calls the LLM to extract
    structured parameter changes as JSON; fills current_value from the base
    config independently.  Persists the draft to results/llm_reviews/{name}/.

    The draft is unapproved (approved=False).  Call approve_experiment_draft()
    before rendering to YAML.
    """
    return generate_draft(
        experiment_name=experiment_name,
        provider=provider,
        model=model,
        proposal_hash=proposal_hash,
        base=base,
        llm_base=llm_base,
        configs_base=configs_base,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def validate_experiment_draft(
    draft: ExperimentDraft,
    base: Path | str | None = None,
    configs_base: Path | str | None = None,
) -> DraftValidationResult:
    """Validate a draft against the experiment config schema.

    Delegates to validate_ml_config() — the authoritative validator.
    No side-effects; returns a DraftValidationResult with is_valid and errors.
    """
    return validate_draft(draft, base=base, configs_base=configs_base)


def approve_experiment_draft(draft: ExperimentDraft) -> ExperimentDraft:
    """Mark a draft as approved, recomputing its hash.

    Returns a new ExperimentDraft with approved=True.  The researcher is
    responsible for validating before approving.  Only approved drafts can
    be rendered to YAML.
    """
    return approve_draft(draft)


def render_draft_to_yaml(
    draft: ExperimentDraft,
    configs_base: Path | str | None = None,
    dry_run: bool = False,
) -> str:
    """Render an approved ExperimentDraft to a YAML config file.

    Requires draft.approved == True.  Applies changes, normalizes, validates,
    hashes for provenance, and writes to configs/experiments/{proposed_name}.yaml.
    Returns the YAML string.  If dry_run=True, skips writing to disk.
    """
    return render_to_yaml(draft, configs_base=configs_base, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Research session (Phase 4)
# ---------------------------------------------------------------------------


def create_research_session(
    root_experiment: str,
    research_goal: str,
    session_id: str | None = None,
    sessions_base: Path | str | None = None,
) -> ResearchSession:
    """Create and persist a new ResearchSession.

    The session tracks a focused research investigation: the goal, the active
    experiment, the event timeline, and the current draft state.

    The session layer is optional and non-intrusive — existing Research API
    functions are not session-aware and do not need to be called through the
    session object.
    """
    return _create_session(
        root_experiment=root_experiment,
        research_goal=research_goal,
        session_id=session_id,
        sessions_base=sessions_base,
    )


def load_research_session(
    session_id: str,
    sessions_base: Path | str | None = None,
) -> ResearchSession | None:
    """Load a ResearchSession from disk.  Returns None if not found."""
    return _load_session(session_id, sessions_base)


def record_session_event(
    session: ResearchSession,
    event_type: str,
    experiment_name: str,
    data: dict | None = None,
    sessions_base: Path | str | None = None,
) -> ResearchSession:
    """Append a SessionEvent to the session and persist.

    Call this after each Research API action to record the event in the
    session timeline.  The session is not automatically updated by other
    API functions — recording events is an explicit researcher action.

    Side effects:
        DRAFT_GENERATED  → session.active_draft_id set from data["draft_id"]
        YAML_RENDERED    → session.active_draft_id cleared
        EXPERIMENT_LINKED → session.active_experiment set from data["new_experiment"]
    """
    return _record_event(
        session=session,
        event_type=event_type,
        experiment_name=experiment_name,
        data=data,
        sessions_base=sessions_base,
    )


def update_research_session_status(
    session: ResearchSession,
    status: str,
    sessions_base: Path | str | None = None,
) -> ResearchSession:
    """Update session status (active / paused / complete) and persist.

    No transition rules are enforced.
    """
    return _update_session_status(session, status, sessions_base)


def summarize_research_session(session: ResearchSession) -> dict:
    """Project a ResearchSession into a flat summary dict.

    Pure computation over the in-memory event log.  No disk I/O.
    Returns a dict suitable for frontend display.
    """
    return _summarize_session(session)


def list_research_sessions(sessions_base: Path | str | None = None) -> list[str]:
    """Return session IDs for all sessions present on disk."""
    return list_session_ids(sessions_base)


def build_research_evolution_chain(
    root_experiment: str,
    base: Path | str | None = None,
    comparisons_base: Path | str | None = None,
    evolution_base: Path | str | None = None,
    persist: bool = True,
) -> ResearchEvolutionChain:
    """Build a research evolution chain rooted at one experiment.

    Follows lineage links from root through all registered descendants,
    derives diagnostics-grounded EvolutionSteps for each transition, and
    assembles a deterministic ResearchEvolutionChain summary.

    No LLM call is made — evolution summaries are computed from existing
    diagnostic artefacts (lineage.json, comparative_review.json, LLMContexts).

    The researcher remains the scientific authority: lineage is human-authored,
    and the chain exposes research evolution rather than autonomous optimization.
    """
    chain = build_evolution_chain(
        root_experiment=root_experiment,
        experiments_base=base,
        comparisons_base=comparisons_base,
    )
    if persist:
        persist_evolution_chain(chain, evolution_base)
    return chain
