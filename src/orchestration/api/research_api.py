"""Top-level structured research API.

This is the single entry-point for the orchestration layer.  All calls from
external code (LLM front-ends, notebooks, CLI tools) should go through this
module.  It composes retrieval, context building, and LLM review into a
clean, typed interface.

The quantitative engine is NEVER called from here, with ONE governed exception:
``execute_approved_config`` (and its ``execute_and_review_*`` composition) runs
a single, already-approved YAML config through the existing engine.  That path
is only ever reached after an explicit, human-authorised request — the advisory
layer never invokes it on its own, never loops, and never runs more than one
config per call.  Everything else only reads persisted artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from src.orchestration.config_generation.draft_generator import (
    generate_draft,
    generate_config_change_draft as _generate_config_change_draft,
    generate_parameter_change_draft as _generate_parameter_change_draft,
)
from src.orchestration.config_generation.draft_generator import (
    load_draft,
    save_draft,
)
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
from src.orchestration.llm.llm_interface import DEFAULT_EMBEDDING_MODEL, embed_texts
from src.orchestration.memory.memory_indexer import build_memory_records
from src.orchestration.memory.memory_retriever import retrieve_memory
from src.orchestration.memory.memory_store import (
    index_exists as _memory_index_exists,
)
from src.orchestration.memory.memory_store import (
    load_records as _load_memory_records,
)
from src.orchestration.memory.memory_store import (
    write_records as _write_memory_records,
)
from src.orchestration.memory.semantic_indexer import build_semantic_index
from src.orchestration.memory.semantic_retriever import rank_by_vector
from src.orchestration.memory.semantic_store import (
    load_manifest as _load_semantic_manifest,
)
from src.orchestration.memory.semantic_store import (
    load_semantic_records as _load_semantic_records,
)
from src.orchestration.memory.semantic_store import (
    semantic_index_exists as _semantic_index_exists,
)
from src.orchestration.session.session_schema import ResearchSession
from src.orchestration.utils.filesystem import (
    comparison_evidence_json_path,
    list_session_ids,
    memory_index_path,
    semantic_memory_index_path,
)

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
    base_url: str | None = None,
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
        base_url=base_url,
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
    reports_base: Path | str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    base_url: str | None = None,
) -> ExperimentDraft:
    """Generate a typed ExperimentDraft from the most recent IterationProposal.

    Requires an IterationProposal already persisted for the experiment
    (run generate_iteration_proposal() first).  Calls the LLM to extract
    structured parameter changes as JSON; fills current_value from the base
    config independently.  Persists the draft to results/llm_reviews/{name}/.

    The proposed name is collision-checked against existing configs, results,
    registry entries, and reports; an already-used name is bumped to the next
    free ``_vN`` suffix so a new draft never silently overwrites prior artefacts.

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
        reports_base=reports_base,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )


def generate_parameter_change_draft(
    experiment_name: str,
    field_path: str,
    proposed_value: Any,
    reason: str | None = None,
    base: Path | str | None = None,
    llm_base: Path | str | None = None,
    configs_base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> dict[str, Any]:
    """Create a draft from one explicit, user-requested config change (no LLM).

    Deterministic: applies exactly ``field_path -> proposed_value`` against the
    experiment's base config, reads the current value from that config, assigns
    the next free ``_vN`` name, validates against the authoritative schema, and
    persists an unapproved draft.  Returns a status dict (``status`` is "ok",
    "config_not_found", "not_ml_config", "invalid_field_path", or
    "schema_incompatible").  Never approves, renders, executes, or calls an LLM.
    """
    return _generate_parameter_change_draft(
        experiment_name=experiment_name,
        field_path=field_path,
        proposed_value=proposed_value,
        reason=reason,
        base=base,
        llm_base=llm_base,
        configs_base=configs_base,
        reports_base=reports_base,
    )


def generate_config_change_draft(
    experiment_name: str,
    changes: list[dict[str, Any]],
    reason: str | None = None,
    base: Path | str | None = None,
    llm_base: Path | str | None = None,
    configs_base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> dict[str, Any]:
    """Create a draft from one or more explicit, schema-validated config changes (no LLM).

    Accepts a list of change dicts with ``field_path``, ``operation``, and ``value``
    (plus ``old_value`` for feature replace).  Supported operations:

    - ``set``     — any whitelisted non-feature field
    - ``add``     — features.entries (value = {name, type, params})
    - ``remove``  — features.entries (value = feature_name_string)
    - ``replace`` — features.entries (old_value = name_to_remove, value = new entry)

    All changes are validated before anything is created.  Returns a status dict
    (``status`` is "ok", "config_not_found", "not_ml_config", "invalid_changes", or
    "schema_incompatible").  Never approves, renders, executes, or calls an LLM.
    """
    return _generate_config_change_draft(
        experiment_name=experiment_name,
        changes=changes,
        reason=reason,
        base=base,
        llm_base=llm_base,
        configs_base=configs_base,
        reports_base=reports_base,
    )


# ---------------------------------------------------------------------------
# Config introspection (read-only, no LLM, no execution)
# ---------------------------------------------------------------------------


# Compact descriptor per changeable field.  Used by list_changeable_config_fields.
# Stays in sync with _VALID_CHANGE_PATHS in draft_validator.py.
def _build_change_surface(
    base_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the controlled change surface with optional current_values."""
    from src.experiments.ml_config import (
        get_valid_label_types,
        get_valid_model_types,
        get_valid_prediction_normalizations,
        get_valid_signal_types,
        get_valid_weighting_schemes,
    )

    def _cur(section: str, *parts: str) -> Any:
        if base_config is None:
            return None
        node: Any = base_config.get(section, {})
        for p in parts:
            if not isinstance(node, dict):
                return None
            node = node.get(p)
        return node

    surface = [
        {
            "field_path": "model.type",
            "type": "str",
            "operations": ["set"],
            "allowed_values": sorted(get_valid_model_types()),
            "description": "ML model algorithm",
            "current_value": _cur("model", "type"),
        },
        {
            "field_path": "model.params.alpha",
            "type": "float",
            "operations": ["set"],
            "description": "L2 regularisation strength (Ridge / Lasso / ElasticNet)",
            "current_value": _cur("model", "params", "alpha"),
        },
        {
            "field_path": "model.params.C",
            "type": "float",
            "operations": ["set"],
            "description": "Inverse regularisation strength (LogisticRegression)",
            "current_value": _cur("model", "params", "C"),
        },
        {
            "field_path": "model.params.l1_ratio",
            "type": "float",
            "operations": ["set"],
            "description": "L1/L2 mixing ratio for ElasticNet (0=Ridge, 1=Lasso)",
            "current_value": _cur("model", "params", "l1_ratio"),
        },
        {
            "field_path": "model.params.max_iter",
            "type": "int",
            "operations": ["set"],
            "description": "Maximum solver iterations",
            "current_value": _cur("model", "params", "max_iter"),
        },
        {
            "field_path": "labels.type",
            "type": "str",
            "operations": ["set"],
            "allowed_values": sorted(get_valid_label_types()),
            "description": "Prediction target type",
            "current_value": _cur("labels", "type"),
        },
        {
            "field_path": "labels.params.horizon",
            "type": "int",
            "operations": ["set"],
            "description": "Forward return horizon in trading days",
            "current_value": _cur("labels", "params", "horizon"),
        },
        {
            "field_path": "signal.type",
            "type": "str",
            "operations": ["set"],
            "allowed_values": sorted(get_valid_signal_types()),
            "description": "Prediction-to-position mapping",
            "current_value": _cur("signal", "type"),
        },
        {
            "field_path": "signal.params.n",
            "type": "int",
            "operations": ["set"],
            "description": "Top-N positions (signal.type=top_n)",
            "current_value": _cur("signal", "params", "n"),
        },
        {
            "field_path": "signal.params.n_long",
            "type": "int",
            "operations": ["set"],
            "description": "Number of long positions (signal.type=long_short)",
            "current_value": _cur("signal", "params", "n_long"),
        },
        {
            "field_path": "signal.params.n_short",
            "type": "int",
            "operations": ["set"],
            "description": "Number of short positions (signal.type=long_short)",
            "current_value": _cur("signal", "params", "n_short"),
        },
        {
            "field_path": "signal.params.threshold",
            "type": "float",
            "operations": ["set"],
            "description": "Threshold for threshold-type signal",
            "current_value": _cur("signal", "params", "threshold"),
        },
        {
            "field_path": "validation.parameters.train_months",
            "type": "int",
            "operations": ["set"],
            "description": "Walk-forward training window in months",
            "current_value": _cur("validation", "parameters", "train_months"),
        },
        {
            "field_path": "validation.parameters.test_months",
            "type": "int",
            "operations": ["set"],
            "description": "Walk-forward test window in months",
            "current_value": _cur("validation", "parameters", "test_months"),
        },
        {
            "field_path": "validation.parameters.gap_days",
            "type": "int",
            "operations": ["set"],
            "description": "Gap days between training and test windows",
            "current_value": _cur("validation", "parameters", "gap_days"),
        },
        {
            "field_path": "execution.transaction_cost_bps",
            "type": "float",
            "operations": ["set"],
            "description": "Round-trip transaction cost in basis points",
            "current_value": _cur("execution", "transaction_cost_bps"),
        },
        {
            "field_path": "portfolio_construction.weighting.scheme",
            "type": "str",
            "operations": ["set"],
            "allowed_values": sorted(get_valid_weighting_schemes()),
            "description": "Portfolio weighting scheme",
            "current_value": _cur("portfolio_construction", "weighting", "scheme"),
        },
        {
            "field_path": "portfolio_construction.weighting.prediction_normalization",
            "type": "str",
            "operations": ["set"],
            "allowed_values": sorted(get_valid_prediction_normalizations()),
            "description": "Prediction normalisation mode",
            "current_value": _cur(
                "portfolio_construction", "weighting", "prediction_normalization"
            ),
        },
        {
            "field_path": "portfolio_construction.weighting.temperature",
            "type": "float|null",
            "operations": ["set"],
            "description": "Softmax temperature for zscore_softmax weighting",
            "current_value": _cur("portfolio_construction", "weighting", "temperature"),
        },
        {
            "field_path": "features.entries",
            "type": "feature_entry",
            "operations": ["add", "remove", "replace"],
            "description": (
                "Feature list. add: {name, type, params}; "
                "remove: feature_name_string; "
                "replace: old_value=feature_to_remove + value={name, type, params}"
            ),
            "current_value": None,
        },
    ]
    # Drop current_value when no config was provided (keep it clean).
    if base_config is None:
        for item in surface:
            del item["current_value"]
    return surface


def inspect_experiment_config(
    experiment_name: str,
    configs_base: Path | str | None = None,
) -> dict[str, Any]:
    """Return a compact summary of the current experiment's YAML config.

    Reads ``configs/experiments/{name}.yaml`` (or the ``configs_base`` override).
    Falls back to the result-directory ``config.json`` if no YAML exists.
    Returns the model, features, validation, and changeable paths — never the
    full raw YAML.  Read-only; no LLM, no execution.

    Returns a status dict:
        {"status": "ok", "experiment_name": ..., "config_path": ...,
         "model": {...}, "features": {...}, "validation": {...},
         "execution": {...}, "labels": {...}, "signal": {...},
         "universe": {...}, "changeable_paths": [...]}
        {"status": "config_not_found", "errors": [...]}
    """
    from src.experiments.config_io import load_config
    from src.orchestration.utils.filesystem import (
        config_path as result_config_path,
        experiment_config_path,
    )
    from src.orchestration.utils.serialization import load_json

    yaml_path = experiment_config_path(experiment_name, configs_base)
    cfg: dict[str, Any] | None = None
    cfg_path_used: str = str(yaml_path)

    if yaml_path.exists():
        try:
            cfg = load_config(yaml_path)
        except Exception:
            cfg = None

    if cfg is None:
        # Try result-directory config.json as fallback.
        result_cfg_p = result_config_path(experiment_name)
        if result_cfg_p.exists():
            cfg = load_json(result_cfg_p)
            cfg_path_used = str(result_cfg_p)

    if cfg is None:
        return {
            "status": "config_not_found",
            "experiment_name": experiment_name,
            "errors": [
                f"No config found for {experiment_name!r}. "
                f"Looked for {yaml_path} and result config.json."
            ],
        }

    entries = cfg.get("features", {}).get("entries", [])
    feature_names = [
        e.get("name") for e in entries if isinstance(e, dict) and e.get("name")
    ]

    surface = _build_change_surface(cfg)
    changeable_paths = [f["field_path"] for f in surface]

    return {
        "status": "ok",
        "experiment_name": experiment_name,
        "config_path": cfg_path_used,
        "model": cfg.get("model", {}),
        "labels": cfg.get("labels", {}),
        "signal": cfg.get("signal", {}),
        "features": {
            "count": len(entries),
            "ticker": cfg.get("features", {}).get("ticker"),
            "entries": feature_names,
        },
        "validation": cfg.get("validation", {}),
        "execution": cfg.get("execution", {}),
        "universe": cfg.get("universe", {}),
        "changeable_paths": changeable_paths,
    }


def list_changeable_config_fields(
    experiment_name: str | None = None,
    configs_base: Path | str | None = None,
) -> dict[str, Any]:
    """Return the controlled config-change surface: paths, operations, and types.

    When ``experiment_name`` is given, the current value for each non-feature
    field is included in the response.  The list is authoritative — only paths
    from this response are accepted by ``generate_config_change_draft``.
    Read-only; no LLM, no execution.

    Returns:
        {"status": "ok", "fields": [...], "feature_entry_count": int|None}
    """
    from src.experiments.config_io import load_config
    from src.orchestration.utils.filesystem import experiment_config_path

    base_config: dict[str, Any] | None = None
    feature_entry_count: int | None = None

    if experiment_name:
        yaml_path = experiment_config_path(experiment_name, configs_base)
        if yaml_path.exists():
            try:
                base_config = load_config(yaml_path)
                entries = base_config.get("features", {}).get("entries", [])
                feature_entry_count = len(entries)
            except Exception:
                base_config = None

    surface = _build_change_surface(base_config)
    return {
        "status": "ok",
        "experiment_name": experiment_name,
        "fields": surface,
        "feature_entry_count": feature_entry_count,
    }


def list_available_features(
    experiment_name: str | None = None,
    family: str | None = None,
    query: str | None = None,
    configs_base: Path | str | None = None,
) -> dict[str, Any]:
    """Return valid feature types from the authoritative schema, with family info.

    When ``experiment_name`` is given, each type is annotated with whether it
    is currently used in that experiment's config.  Optional ``family`` filter
    (case-insensitive substring) and ``query`` filter (type name / family
    substring).  Returns only schema-validated types — never invents names.
    Read-only; no LLM, no execution.

    Returns:
        {"status": "ok", "features": [...], "total_types": int, "currently_used_count": int|None}
    """
    from src.experiments.config_io import load_config
    from src.experiments.ml_config import get_feature_required_params, get_valid_feature_types
    from src.features.families import get_family_for_type
    from src.orchestration.utils.filesystem import experiment_config_path

    valid_types = sorted(get_valid_feature_types())
    req_params = get_feature_required_params()

    # Collect used types from the experiment config if given.
    used_types: set[str] = set()
    used_names: set[str] = set()
    currently_used_count: int | None = None
    if experiment_name:
        yaml_path = experiment_config_path(experiment_name, configs_base)
        if yaml_path.exists():
            try:
                cfg = load_config(yaml_path)
                for e in cfg.get("features", {}).get("entries", []):
                    if isinstance(e, dict):
                        if e.get("type"):
                            used_types.add(e["type"])
                        if e.get("name"):
                            used_names.add(e["name"])
                currently_used_count = len(used_types)
            except Exception:
                pass

    features: list[dict[str, Any]] = []
    for ftype in valid_types:
        feat_family = get_family_for_type(ftype)

        # Apply family filter (case-insensitive substring).
        if family and family.lower() not in feat_family.lower():
            continue
        # Apply query filter (substring on type name or family name).
        if query:
            ql = query.lower()
            if ql not in ftype.lower() and ql not in feat_family.lower():
                continue

        features.append({
            "type": ftype,
            "family": feat_family,
            "required_params": sorted(req_params.get(ftype, frozenset())),
            "currently_used": ftype in used_types if experiment_name else None,
            "operations": ["add", "remove", "replace"],
        })

    return {
        "status": "ok",
        "experiment_name": experiment_name,
        "features": features,
        "total_types": len(features),
        "currently_used_count": currently_used_count,
        "current_feature_names": sorted(used_names) if experiment_name else None,
    }


def list_supported_models(
    experiment_name: str | None = None,
    configs_base: Path | str | None = None,
) -> dict[str, Any]:
    """Return model types supported by the quant engine / config schema.

    When ``experiment_name`` is given, the current model type is highlighted.
    Model switching IS supported — ``generate_config_change_draft`` with
    ``operation="set"`` on ``model.type`` accepts any value in ``supported_models``.
    Read-only; no LLM, no execution; never invents model names.

    Returns:
        {"status": "ok", "current_model": str|None, "supported_models": [...],
         "model_switching_supported": bool}
    """
    from src.experiments.config_io import load_config
    from src.experiments.ml_config import get_valid_model_types
    from src.orchestration.utils.filesystem import experiment_config_path

    valid_models = sorted(get_valid_model_types())

    # Required and optional params per model type (static knowledge).
    _MODEL_PARAMS: dict[str, dict[str, list[str]]] = {
        "LinearRegression":     {"required": [], "optional": []},
        "RidgeRegression":      {"required": [], "optional": ["alpha"]},
        "LassoRegression":      {"required": [], "optional": ["alpha", "max_iter"]},
        "ElasticNetRegression": {"required": [], "optional": ["alpha", "l1_ratio", "max_iter"]},
        "LogisticRegression":   {"required": [], "optional": ["C", "max_iter"]},
    }

    current_model: str | None = None
    if experiment_name:
        yaml_path = experiment_config_path(experiment_name, configs_base)
        if yaml_path.exists():
            try:
                cfg = load_config(yaml_path)
                current_model = cfg.get("model", {}).get("type")
            except Exception:
                pass

    supported = [
        {
            "name": m,
            "required_params": _MODEL_PARAMS.get(m, {}).get("required", []),
            "optional_params": _MODEL_PARAMS.get(m, {}).get("optional", []),
            "is_current": (m == current_model),
        }
        for m in valid_models
    ]

    return {
        "status": "ok",
        "experiment_name": experiment_name,
        "current_model": current_model,
        "supported_models": supported,
        "model_switching_supported": True,
    }


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


def approve_experiment_draft(
    draft: ExperimentDraft,
    llm_base: Path | str | None = None,
) -> ExperimentDraft:
    """Mark a draft as approved, recomputing its hash, and persist it.

    Returns a new ExperimentDraft with approved=True.  The approved draft is
    written back to disk so a later, stateless ``load_experiment_draft`` (e.g.
    from a separate API/MCP call) sees the approval before rendering.  The
    researcher is responsible for validating before approving.  Only approved
    drafts can be rendered to YAML.
    """
    approved = approve_draft(draft)
    save_draft(approved, llm_base)
    return approved


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


def get_latest_research_session(
    sessions_base: Path | str | None = None,
) -> ResearchSession | None:
    """Return the most recently updated ResearchSession, or None if none exist.

    Read-only recovery helper: lets a caller that has lost the session_id find
    the active session again.  Loads persisted sessions and returns the one with
    the latest ``updated_at``.  Never raises.
    """
    sessions = [
        s
        for sid in list_session_ids(sessions_base)
        if (s := _load_session(sid, sessions_base)) is not None
    ]
    if not sessions:
        return None
    return max(sessions, key=lambda s: s.updated_at)


# ---------------------------------------------------------------------------
# Research memory (Phase 1 — metadata/keyword RAG)
#
# A lightweight, controlled evidence layer.  It indexes compact summaries of
# existing artefacts into a local JSONL file and retrieves prior research
# evidence by deterministic keyword / metadata matching.  It is evidence-only:
# it runs no experiment, calls no LLM, approves nothing, renders nothing, and
# never authorises execution.  Full artefacts stay on disk.
# ---------------------------------------------------------------------------


def get_research_memory_status(
    memory_base: Path | str | None = None,
) -> dict[str, Any]:
    """Report whether the memory index exists and how much it holds (read-only)."""
    records = _load_memory_records(memory_base)
    experiment_count = len({r.experiment_name for r in records if r.experiment_name})
    return {
        "index_exists": _memory_index_exists(memory_base),
        "item_count": len(records),
        "experiment_count": experiment_count,
        "index_path": str(memory_index_path(memory_base)),
    }


def index_research_memory(
    base: Path | str | None = None,
    llm_base: Path | str | None = None,
    reports_base: Path | str | None = None,
    sessions_base: Path | str | None = None,
    memory_base: Path | str | None = None,
    comparisons_base: Path | str | None = None,
) -> dict[str, Any]:
    """Build or refresh the research-memory index from known artefact locations.

    Controlled write: reads only known Zeto artefact paths and overwrites the
    local JSONL index with compact records.  Runs no experiment, calls no LLM,
    approves/renders/executes nothing, and never mutates the source artefacts.
    """
    records = build_memory_records(
        base=base,
        llm_base=llm_base,
        reports_base=reports_base,
        sessions_base=sessions_base,
        comparisons_base=comparisons_base,
    )
    path = _write_memory_records(records, memory_base)
    experiment_count = len({r.experiment_name for r in records if r.experiment_name})
    return {
        "indexed_count": len(records),
        "experiment_count": experiment_count,
        "index_path": str(path),
    }


def retrieve_research_memory(
    query: str | None = None,
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    top_k: int = 5,
    memory_base: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve prior research evidence by deterministic keyword/metadata match.

    Read-only.  Returns up to ``top_k`` compact items (summaries, paths, hashes,
    tags, matched terms) — never full artefact contents.  Evidence only: the
    result authorises no execution, approval, or rendering.
    """
    records = _load_memory_records(memory_base)
    return retrieve_memory(
        records,
        query=query,
        experiment_name=experiment_name,
        failure_modes=failure_modes,
        artefact_type=artefact_type,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Research memory (Phase 2 — local semantic retrieval)
#
# Layers a local embedding index on top of the Phase 1 records (the source of
# truth).  Embeds compact summaries only, via a local OpenAI-compatible
# embedding endpoint — never a chat/completion model.  Still evidence-only: it
# runs no experiment, approves nothing, renders nothing, and authorises no
# execution.  Full artefacts stay on disk.
# ---------------------------------------------------------------------------


def get_semantic_research_memory_status(
    memory_base: Path | str | None = None,
) -> dict[str, Any]:
    """Report whether the semantic index exists and how much it holds (read-only)."""
    records = _load_semantic_records(memory_base)
    manifest = _load_semantic_manifest(memory_base) or {}
    model = manifest.get("embedding_model") or (
        records[0].embedding_model if records else DEFAULT_EMBEDDING_MODEL
    )
    return {
        "index_exists": _semantic_index_exists(memory_base),
        "item_count": len(records),
        "embedding_model": model,
        "embedding_dim": manifest.get("embedding_dim")
        or (records[0].embedding_dim if records else 0),
        "index_path": str(semantic_memory_index_path(memory_base)),
    }


def index_semantic_research_memory(
    provider: str = "openai",
    model: str | None = None,
    base_url: str | None = None,
    memory_base: Path | str | None = None,
) -> dict[str, Any]:
    """Build/refresh the semantic index by embedding Phase 1 memory records.

    Controlled write.  Reads only the Phase 1 JSONL index, embeds compact
    summaries via the embeddings endpoint (no chat/LLM, no experiment, no
    approval/render/execution), reuses unchanged embeddings, and overwrites the
    semantic index in place.  Returns a status dict whose ``status`` is one of
    "ok", "no_phase1_index", or "embedding_failed".
    """
    return build_semantic_index(
        provider=provider,
        model=model,
        base_url=base_url,
        memory_base=memory_base,
    )


def semantic_retrieve_research_memory(
    query: str,
    top_k: int = 5,
    experiment_name: str | None = None,
    failure_modes: list[str] | None = None,
    artefact_type: str | None = None,
    tags: list[str] | None = None,
    provider: str = "openai",
    model: str | None = None,
    base_url: str | None = None,
    memory_base: Path | str | None = None,
) -> dict[str, Any]:
    """Retrieve prior research evidence by local semantic similarity.

    Read-only.  Embeds the query with the same local embedding model, ranks the
    semantic index by cosine similarity, applies optional metadata filters, and
    returns up to ``top_k`` compact items (scores, paths, hashes, tags, failure
    modes, summaries) — never full artefact bodies.

    Returns a status dict whose ``status`` is one of "ok", "no_phase1_index",
    "no_semantic_index", or "embedding_failed".  Never auto-retries; never
    invents evidence.
    """
    if not _memory_index_exists(memory_base):
        return {"status": "no_phase1_index", "query": query, "items": []}
    if not _semantic_index_exists(memory_base):
        return {"status": "no_semantic_index", "query": query, "items": []}

    records = _load_semantic_records(memory_base)
    manifest = _load_semantic_manifest(memory_base) or {}
    # Match the query model to the index model for a meaningful comparison.
    resolved_model = model or manifest.get("embedding_model") or DEFAULT_EMBEDDING_MODEL

    try:
        resp = embed_texts(
            [query], provider=provider, model=resolved_model, base_url=base_url
        )
    except Exception as exc:  # noqa: BLE001 — clean envelope, never a traceback
        return {
            "status": "embedding_failed",
            "query": query,
            "items": [],
            "error": f"{type(exc).__name__}: {exc}",
            "embedding_model": resolved_model,
        }

    query_vector = resp.vectors[0] if resp.vectors else []
    items = rank_by_vector(
        records,
        query_vector,
        experiment_name=experiment_name,
        failure_modes=failure_modes,
        artefact_type=artefact_type,
        tags=tags,
        top_k=top_k,
    )
    return {
        "status": "ok",
        "query": query,
        "items": items,
        "embedding_model": resolved_model,
    }


# ---------------------------------------------------------------------------
# Authoritative experiment metrics (read from artefacts, NOT from RAG memory)
#
# These read the deterministic on-disk diagnostics (metrics.json, split metrics,
# failure modes) via the same context builder used elsewhere — never from the
# research-memory index.  No LLM, no execution, no arbitrary-path parsing.
# ---------------------------------------------------------------------------

# Authoritative metric fields, grouped by their context source.
_PERF_METRIC_KEYS = (
    "sharpe_ratio",
    "annualized_return_pct",
    "annualized_volatility_pct",
    "max_drawdown_pct",
    "calmar_ratio",
    "hit_rate_pct",
)
_VAL_METRIC_KEYS = (
    "mean_oos_sharpe",
    "std_oos_sharpe",
    "n_splits",
    "n_negative_sharpe_splits",
    "hit_rate_positive_sharpe_pct",
    "worst_split_drawdown_pct",
    "consistency_tier",
)


def get_experiment_metrics(
    experiment_name: str,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> dict[str, Any]:
    """Load authoritative metrics for one experiment from its artefacts.

    Reads only known locations (metrics.json + validation split diagnostics) via
    the deterministic context builder.  Returns a status dict with compact
    metrics, a ``missing_metrics`` list (never invented values), detected failure
    modes, and the metrics/report/plots paths.  ``status`` is "ok" or
    "not_found".  No LLM, no execution, no RAG memory.
    """
    from src.orchestration.utils.filesystem import (
        metadata_path,
        metrics_path,
        plots_dir,
        report_markdown_path,
    )

    metrics_p = metrics_path(experiment_name, base)
    if not (metrics_p.exists() or metadata_path(experiment_name, base).exists()):
        all_keys = list(_PERF_METRIC_KEYS) + list(_VAL_METRIC_KEYS)
        return {
            "status": "not_found",
            "experiment_name": experiment_name,
            "metrics": {k: None for k in all_keys},
            "missing_metrics": all_keys,
            "failure_modes": [],
            "metrics_path": None,
            "report_path": None,
            "plots_dir": None,
        }

    ctx = build_context(experiment_name, base=base, reports_base=reports_base)
    perf = ctx.performance or {}
    val = ctx.validation or {}

    metrics: dict[str, Any] = {}
    missing: list[str] = []
    for key in _PERF_METRIC_KEYS:
        metrics[key] = perf.get(key)
        if metrics[key] is None:
            missing.append(key)
    for key in _VAL_METRIC_KEYS:
        metrics[key] = val.get(key)
        if metrics[key] is None:
            missing.append(key)

    failure_modes = [fm.get("name") for fm in ctx.failure_modes if fm.get("name")]

    report_p = report_markdown_path(experiment_name, reports_base)
    plots_p = plots_dir(experiment_name, base)
    return {
        "status": "ok",
        "experiment_name": experiment_name,
        "metrics": metrics,
        "missing_metrics": missing,
        "failure_modes": failure_modes,
        "metrics_path": str(metrics_p) if metrics_p.exists() else None,
        "report_path": str(report_p) if report_p.exists() else None,
        "plots_dir": str(plots_p) if plots_p.exists() else None,
    }


def compare_experiment_metrics(
    base_experiment_name: str,
    candidate_experiment_name: str,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
    comparisons_base: Path | str | None = None,
    session_id: str | None = None,
    research_question: str | None = None,
    tested_change: str | None = None,
    persist_evidence: bool = True,
) -> dict[str, Any]:
    """Compare two experiments using authoritative artefact metrics only.

    Returns a status dict with base/candidate/delta for Sharpe, mean OOS Sharpe,
    and max drawdown (deltas only where both values are numeric), each
    experiment's failure modes, and a concise conclusion.  ``status`` is "ok" or
    "not_found" (with which experiments are missing).  No LLM, no RAG memory.

    When ``persist_evidence=True`` (default), a compact ``comparison_evidence``
    JSON record is written to
    ``results/comparisons/<base>__vs__<candidate>/comparison_evidence.json``
    so that Phase 1 and Phase 2 memory indexers can retrieve before/after
    conclusions without re-reading full artefacts.
    """
    base_res = get_experiment_metrics(base_experiment_name, base=base, reports_base=reports_base)
    cand_res = get_experiment_metrics(candidate_experiment_name, base=base, reports_base=reports_base)

    missing_experiments = [
        name
        for name, res in (
            (base_experiment_name, base_res),
            (candidate_experiment_name, cand_res),
        )
        if res["status"] == "not_found"
    ]
    if missing_experiments:
        return {
            "status": "not_found",
            "base_experiment": base_experiment_name,
            "candidate_experiment": candidate_experiment_name,
            "missing_experiments": missing_experiments,
        }

    bm = base_res["metrics"]
    cm = cand_res["metrics"]

    result = {
        "status": "ok",
        "base_experiment": base_experiment_name,
        "candidate_experiment": candidate_experiment_name,
        "base_sharpe": bm.get("sharpe_ratio"),
        "candidate_sharpe": cm.get("sharpe_ratio"),
        "delta_sharpe": _num_delta(bm.get("sharpe_ratio"), cm.get("sharpe_ratio")),
        "base_mean_oos_sharpe": bm.get("mean_oos_sharpe"),
        "candidate_mean_oos_sharpe": cm.get("mean_oos_sharpe"),
        "delta_mean_oos_sharpe": _num_delta(
            bm.get("mean_oos_sharpe"), cm.get("mean_oos_sharpe")
        ),
        "base_max_drawdown": bm.get("max_drawdown_pct"),
        "candidate_max_drawdown": cm.get("max_drawdown_pct"),
        "delta_max_drawdown_pct": _num_delta(
            _pct_to_float(bm.get("max_drawdown_pct")),
            _pct_to_float(cm.get("max_drawdown_pct")),
        ),
        "base_failure_modes": base_res["failure_modes"],
        "candidate_failure_modes": cand_res["failure_modes"],
    }
    result["conclusion"] = _comparison_conclusion(result)

    if persist_evidence:
        evidence_path = _persist_comparison_evidence(
            result=result,
            base_res=base_res,
            cand_res=cand_res,
            comparisons_base=comparisons_base,
            session_id=session_id,
            research_question=research_question,
            tested_change=tested_change,
        )
        result["evidence_path"] = evidence_path
    else:
        result["evidence_path"] = None

    return result


def _persist_comparison_evidence(
    result: dict[str, Any],
    base_res: dict[str, Any],
    cand_res: dict[str, Any],
    comparisons_base: Path | str | None,
    session_id: str | None,
    research_question: str | None,
    tested_change: str | None,
) -> str | None:
    """Write a compact comparison_evidence JSON record.  Returns the path written or None."""
    from datetime import UTC, datetime

    from src.orchestration.utils.serialization import dump_json

    base_name = result["base_experiment"]
    candidate_name = result["candidate_experiment"]

    evidence = {
        "artefact_type": "comparison_evidence",
        "base_experiment_name": base_name,
        "candidate_experiment_name": candidate_name,
        "research_question": research_question,
        "tested_change": tested_change,
        "session_id": session_id,
        "base_metrics": {
            "sharpe_ratio": result["base_sharpe"],
            "mean_oos_sharpe": result["base_mean_oos_sharpe"],
            "max_drawdown_pct": result["base_max_drawdown"],
        },
        "candidate_metrics": {
            "sharpe_ratio": result["candidate_sharpe"],
            "mean_oos_sharpe": result["candidate_mean_oos_sharpe"],
            "max_drawdown_pct": result["candidate_max_drawdown"],
        },
        "metric_deltas": {
            "delta_sharpe": result["delta_sharpe"],
            "delta_mean_oos_sharpe": result["delta_mean_oos_sharpe"],
            "delta_max_drawdown_pct": result["delta_max_drawdown_pct"],
        },
        "failure_modes_base": result["base_failure_modes"],
        "failure_modes_candidate": result["candidate_failure_modes"],
        "conclusion": result["conclusion"],
        "artefact_paths": {
            "base_metrics": base_res.get("metrics_path"),
            "candidate_metrics": cand_res.get("metrics_path"),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }

    try:
        path = comparison_evidence_json_path(base_name, candidate_name, comparisons_base)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(evidence, path)
        return str(path)
    except Exception:  # noqa: BLE001
        return None


def inspect_comparison_evidence(
    base_experiment_name: str,
    candidate_experiment_name: str,
    comparisons_base: Path | str | None = None,
) -> dict[str, Any]:
    """Return a compact summary of a specific comparison evidence record.

    Reads ``results/comparisons/<base>__vs__<candidate>/comparison_evidence.json``
    only.  No arbitrary path input accepted.  No LLM, no RAG, no execution.

    Returns:
        {"status": "ok", "base_experiment_name": ..., "candidate_experiment_name": ...,
         "tested_change": ..., "metric_deltas": {...}, ...}
        {"status": "not_found", "errors": [...]}
    """
    from src.orchestration.utils.serialization import load_json

    path = comparison_evidence_json_path(
        base_experiment_name, candidate_experiment_name, comparisons_base
    )
    data = load_json(path)

    if not isinstance(data, dict):
        return {
            "status": "not_found",
            "base_experiment_name": base_experiment_name,
            "candidate_experiment_name": candidate_experiment_name,
            "evidence_path": str(path),
            "errors": [
                f"No comparison evidence found for {base_experiment_name!r} vs "
                f"{candidate_experiment_name!r}. "
                "Run compare_experiment_metrics first to create the record."
            ],
        }

    return {
        "status": "ok",
        "base_experiment_name": base_experiment_name,
        "candidate_experiment_name": candidate_experiment_name,
        "research_question": data.get("research_question"),
        "tested_change": data.get("tested_change"),
        "base_metrics": data.get("base_metrics") or {},
        "candidate_metrics": data.get("candidate_metrics") or {},
        "metric_deltas": data.get("metric_deltas") or {},
        "failure_modes_base": list(data.get("failure_modes_base") or []),
        "failure_modes_candidate": list(data.get("failure_modes_candidate") or []),
        "conclusion": data.get("conclusion") or "",
        "session_id": data.get("session_id"),
        "created_at": data.get("created_at") or "",
        "evidence_path": str(path),
    }


def _num_delta(a: Any, b: Any) -> float | None:
    """candidate - base, rounded, only when both are numeric; else None."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(b - a, 3)
    return None


def _pct_to_float(value: Any) -> float | None:
    """Parse a '-31.20%' string into the numeric -31.20 (percentage points)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip().endswith("%"):
        try:
            return float(value.strip()[:-1])
        except ValueError:
            return None
    return None


def _comparison_conclusion(r: dict[str, Any]) -> str:
    """Build a concise, deterministic comparison sentence (no LLM)."""
    def fmt(v: Any) -> str:
        return "n/a" if v is None else str(v)

    parts = [
        f"Comparison {r['base_experiment']} -> {r['candidate_experiment']}: "
        f"Sharpe {fmt(r['base_sharpe'])} -> {fmt(r['candidate_sharpe'])}",
        f"mean OOS Sharpe {fmt(r['base_mean_oos_sharpe'])} -> {fmt(r['candidate_mean_oos_sharpe'])}",
    ]
    cand_modes = r.get("candidate_failure_modes") or []
    if cand_modes:
        parts.append("candidate failure modes: " + ", ".join(cand_modes))
    else:
        parts.append("candidate has no detected failure modes")
    return "; ".join(parts) + "."


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


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------


def compute_context_hash(context: LLMContext) -> str:
    """Public wrapper over the review engine's deterministic context hash.

    Lets callers (demo, API endpoint) record the provenance hash of a context
    without importing a private symbol.
    """
    from src.orchestration.llm.review_engine import _compute_context_hash

    return _compute_context_hash(context)


# ---------------------------------------------------------------------------
# Persisted-artefact path accessors (so compact callers can return references
# to full outputs on disk instead of inlining them)
# ---------------------------------------------------------------------------


def review_artifact_path(experiment_name: str, llm_base: Path | str | None = None) -> str:
    """Path to the persisted LLM review JSON for an experiment."""
    from src.orchestration.utils.filesystem import llm_review_path

    return str(llm_review_path(experiment_name, llm_base))


def proposal_artifact_path(experiment_name: str, llm_base: Path | str | None = None) -> str:
    """Path to the persisted iteration-proposal JSON for an experiment."""
    from src.orchestration.utils.filesystem import iteration_proposal_json_path

    return str(iteration_proposal_json_path(experiment_name, llm_base))


def draft_artifact_path(
    experiment_name: str,
    draft_id: str,
    llm_base: Path | str | None = None,
) -> str:
    """Path to the persisted draft JSON for an experiment + draft id."""
    from src.orchestration.utils.filesystem import draft_json_path

    return str(draft_json_path(experiment_name, draft_id, llm_base))


# ---------------------------------------------------------------------------
# Governed execution bridge (Phase 5)
#
# Human-controlled execution only.  The advisory layer never calls these
# functions on its own — they are reached solely through an explicit researcher
# action (a confirmed CLI flag or an authorised API request).  Each call runs
# exactly one already-approved config; there is no loop, no retry, and no
# metric-driven re-run.
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Outcome of running one approved config through the existing engine."""

    config_path: str
    experiment_name: str | None
    success: bool
    artefact_root: str | None
    report_path: str | None
    error: str | None = None
    command_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "experiment_name": self.experiment_name,
            "success": self.success,
            "artefact_root": self.artefact_root,
            "report_path": self.report_path,
            "error": self.error,
            "command_hint": self.command_hint,
        }


@dataclass
class ExecutionReviewResult:
    """Execution plus the LLM review of the freshly generated artefacts.

    ``review`` is None when execution failed, was a dry run, or produced no
    identifiable experiment — post-run review never runs on a failed execution.
    """

    execution: ExecutionResult
    review: LLMReviewOutput | None
    context_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution": self.execution.to_dict(),
            "review": self.review.to_dict() if self.review is not None else None,
            "context_hash": self.context_hash,
        }


def _execution_command_hint(config_path: Path, report: bool, preset: str) -> str:
    cmd = f"python scripts/run_from_config.py {config_path}"
    if report:
        cmd += f" --report --preset {preset}"
    return cmd


def execute_approved_config(
    config_path: str | Path,
    report: bool = True,
    preset: str = "canonical",
    dry_run: bool = False,
    report_output_dir: str | Path | None = None,
) -> ExecutionResult:
    """Run exactly one approved YAML config through the existing engine.

    This reuses the same code path as ``scripts/run_from_config.py`` — it does
    not duplicate or re-implement any experiment-running logic.  The engine
    module is imported lazily so the orchestration import graph stays light and
    so this heavy dependency is only loaded when execution is actually invoked.

    Args:
        config_path:       Path to an approved, rendered YAML config.
        report:            Whether to generate a report (uses ``preset``).
        preset:            Report preset when ``report`` is True.
        dry_run:           If True, return the planned command without running.
        report_output_dir: Optional report output directory.

    Returns:
        ExecutionResult.  ``success=False`` carries a human-readable ``error``;
        this function never raises for an engine/IO failure.
    """
    config_path = Path(config_path)
    command_hint = _execution_command_hint(config_path, report, preset)

    if dry_run:
        return ExecutionResult(
            config_path=str(config_path),
            experiment_name=None,
            success=True,
            artefact_root=None,
            report_path=None,
            error=None,
            command_hint=command_hint,
        )

    if not config_path.exists():
        return ExecutionResult(
            config_path=str(config_path),
            experiment_name=None,
            success=False,
            artefact_root=None,
            report_path=None,
            error=f"Config not found: {config_path}",
            command_hint=command_hint,
        )

    try:
        # Lazy import — same engine entry points used by run_from_config.py.
        from src.experiments.orchestrator import (
            run_and_report,
            run_experiment_from_config,
        )

        if report:
            run, paths = run_and_report(
                config_path,
                report_spec=_resolve_report_preset(preset),
                **({"report_output_dir": report_output_dir} if report_output_dir else {}),
            )
            report_path = str(getattr(paths, "markdown", "") or "") or None
        else:
            run = run_experiment_from_config(config_path)
            report_path = None

        artefact_root = Path(run.output_path)
        return ExecutionResult(
            config_path=str(config_path),
            experiment_name=artefact_root.name,
            success=True,
            artefact_root=str(artefact_root),
            report_path=report_path,
            error=None,
            command_hint=command_hint,
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean result, never raise
        return ExecutionResult(
            config_path=str(config_path),
            experiment_name=None,
            success=False,
            artefact_root=None,
            report_path=None,
            error=f"{type(exc).__name__}: {exc}",
            command_hint=command_hint,
        )


def _resolve_report_preset(preset: str) -> Any:
    """Map a preset name to a report spec, mirroring run_from_config.py."""
    from src.reporting.report_spec import (
        AUDIT_REPORT,
        CANONICAL_SHOWCASE,
        COMPACT_REPORT,
        DIAGNOSTICS_REPORT,
        STANDARD_REPORT,
    )

    return {
        "standard": STANDARD_REPORT,
        "canonical": CANONICAL_SHOWCASE,
        "compact": COMPACT_REPORT,
        "diagnostics": DIAGNOSTICS_REPORT,
        "audit": AUDIT_REPORT,
    }.get(preset, CANONICAL_SHOWCASE)


def execute_and_review_approved_config(
    config_path: str | Path,
    provider: str = "stub",
    model: str | None = None,
    base_url: str | None = None,
    report: bool = True,
    preset: str = "canonical",
    dry_run: bool = False,
) -> ExecutionReviewResult:
    """Run one approved config, then LLM-review the freshly generated artefacts.

    Composition only — no new execution or review logic.  Post-run review is
    skipped when execution fails, is a dry run, or yields no experiment name.
    Session-event recording is the caller's responsibility.
    """
    execution = execute_approved_config(
        config_path,
        report=report,
        preset=preset,
        dry_run=dry_run,
    )

    if dry_run or not execution.success or not execution.experiment_name:
        return ExecutionReviewResult(execution=execution, review=None, context_hash=None)

    context = build_llm_context(execution.experiment_name, persist=False)
    context_hash = compute_context_hash(context)
    review = run_llm_review(
        execution.experiment_name,
        provider=provider,
        model=model,
        base_url=base_url,
        persist_context=False,
        persist_review=True,
    )
    return ExecutionReviewResult(
        execution=execution,
        review=review,
        context_hash=context_hash,
    )


# ---------------------------------------------------------------------------
# Workflow state (read-only inspection)
# ---------------------------------------------------------------------------


def get_research_workflow_state(
    experiment_name: str,
    base: Path | str | None = None,
    llm_base: Path | str | None = None,
    configs_base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> dict[str, Any]:
    """Report the on-disk state of one experiment's research workflow.

    Pure read-only inspection — checks only for the presence of artefacts on
    disk.  Executes nothing, calls no LLM, and never touches the quant engine.

    Returned keys (all plain primitives):
        experiment_name, baseline_artefacts_exist, metadata_exists,
        metrics_exists, context_ready, review_exists, proposal_exists,
        draft_exists, latest_draft_id, latest_draft_approved, proposed_name,
        rendered_yaml_exists, rendered_yaml_path, revised_artefacts_exist,
        report_path, plots_dir
    """
    from src.orchestration.utils.filesystem import (
        experiment_config_path,
        experiment_root,
        iteration_proposal_json_path,
        llm_review_dir,
        llm_review_path,
        metadata_path,
        metrics_path,
        plots_dir,
        report_markdown_path,
    )
    from src.orchestration.utils.serialization import load_json

    metadata_exists = metadata_path(experiment_name, base).exists()
    metrics_exists = metrics_path(experiment_name, base).exists()
    context_ready = metadata_exists and metrics_exists
    baseline_artefacts_exist = (
        experiment_root(experiment_name, base).exists() and context_ready
    )

    review_exists = llm_review_path(experiment_name, llm_base).exists()
    proposal_exists = iteration_proposal_json_path(experiment_name, llm_base).exists()

    # Locate the most recent draft (by generated_at) without mutating anything.
    review_dir = llm_review_dir(experiment_name, llm_base)
    latest_draft: dict[str, Any] | None = None
    if review_dir.exists():
        drafts = [d for f in sorted(review_dir.glob("draft_*.json")) if (d := load_json(f))]
        if drafts:
            drafts.sort(key=lambda d: str(d.get("generated_at", "")))
            latest_draft = drafts[-1]

    draft_exists = latest_draft is not None
    latest_draft_id = latest_draft.get("draft_id") if latest_draft else None
    latest_draft_approved = bool(latest_draft.get("approved")) if latest_draft else False
    proposed_name = latest_draft.get("proposed_name") if latest_draft else None

    rendered_yaml_exists = False
    rendered_yaml_path: str | None = None
    revised_artefacts_exist = False
    revised_report: str | None = None
    revised_plots: str | None = None
    if proposed_name:
        cfg = experiment_config_path(proposed_name, configs_base)
        rendered_yaml_exists = cfg.exists()
        rendered_yaml_path = str(cfg) if rendered_yaml_exists else None
        revised_artefacts_exist = metadata_path(proposed_name, base).exists()
        rp = report_markdown_path(proposed_name, reports_base)
        revised_report = str(rp) if rp.exists() else None
        rpl = plots_dir(proposed_name, base)
        revised_plots = str(rpl) if rpl.exists() else None

    base_report = report_markdown_path(experiment_name, reports_base)
    base_report_path = str(base_report) if base_report.exists() else None
    base_plots = plots_dir(experiment_name, base)
    base_plots_dir = str(base_plots) if base_plots.exists() else None

    return {
        "experiment_name": experiment_name,
        "baseline_artefacts_exist": baseline_artefacts_exist,
        "metadata_exists": metadata_exists,
        "metrics_exists": metrics_exists,
        "context_ready": context_ready,
        "review_exists": review_exists,
        "proposal_exists": proposal_exists,
        "draft_exists": draft_exists,
        "latest_draft_id": latest_draft_id,
        "latest_draft_approved": latest_draft_approved,
        "proposed_name": proposed_name,
        "rendered_yaml_exists": rendered_yaml_exists,
        "rendered_yaml_path": rendered_yaml_path,
        "revised_artefacts_exist": revised_artefacts_exist,
        # Prefer the revised experiment's artefacts when present, else baseline.
        "report_path": revised_report or base_report_path,
        "plots_dir": revised_plots or base_plots_dir,
    }
