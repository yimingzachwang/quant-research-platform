"""Structured Research Orchestration Layer.

This package wraps the quantitative research engine with a structured API
designed for LLM-assisted interpretation.  The LLM never touches the quant
engine directly — all communication goes through typed context objects.

Quick start::

    from src.orchestration.api import build_llm_context, run_llm_review

    # Build structured context from persisted artefacts (no quant engine)
    context = build_llm_context("canonical_ml_multi_asset")

    # Run LLM review (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)
    review = run_llm_review("canonical_ml_multi_asset", provider="anthropic")
    print(review.review_text)

    # Or use stub provider for testing
    review = run_llm_review("canonical_ml_multi_asset", provider="stub")
"""

from src.orchestration.api.research_api import (
    approve_experiment_draft,
    build_llm_context,
    compare,
    create_research_session,
    diff,
    find_experiments,
    generate_experiment_draft,
    get_experiment_plots,
    get_experiment_summary,
    list_all_experiments,
    list_artefacts,
    list_experiment_summaries,
    list_research_sessions,
    load_experiment,
    load_experiment_draft,
    load_research_session,
    rank,
    rank_experiments_by_sharpe,
    record_session_event,
    render_draft_to_yaml,
    retrieve_all_diagnostics,
    retrieve_artefact,
    run_llm_review,
    summarize_research_session,
    update_research_session_status,
    validate_experiment_draft,
)

__all__ = [
    "list_all_experiments",
    "find_experiments",
    "get_experiment_summary",
    "list_experiment_summaries",
    "rank_experiments_by_sharpe",
    "load_experiment",
    "retrieve_artefact",
    "retrieve_all_diagnostics",
    "list_artefacts",
    "get_experiment_plots",
    "compare",
    "diff",
    "rank",
    "build_llm_context",
    "run_llm_review",
    "generate_experiment_draft",
    "load_experiment_draft",
    "validate_experiment_draft",
    "approve_experiment_draft",
    "render_draft_to_yaml",
    "create_research_session",
    "load_research_session",
    "record_session_event",
    "update_research_session_status",
    "summarize_research_session",
    "list_research_sessions",
]
