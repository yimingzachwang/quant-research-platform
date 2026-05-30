"""Workflow router: maps typed intents to research API calls.

The router is the only place that calls src.orchestration.api.research_api.
It never calls quantitative engine code directly, never generates config, and
never executes experiments.  Each branch is a thin dispatch to one existing
API function.

Design constraints (from the implementation mandate):
  - DO NOT call quant engine functions
  - DO NOT synthesise experiment configs
  - DO NOT introduce autonomous execution loops
  - Only call functions already present in research_api.py
"""

from __future__ import annotations

import time
from pathlib import Path

import src.orchestration.api.research_api as _api
from src.orchestration.intents.intent_schema import (
    BuildContextIntent,
    BuildEvolutionChainIntent,
    CompareExperimentsIntent,
    GenerateDraftIntent,
    GenerateIterationIntent,
    Intent,
    ListExperimentsIntent,
    RankExperimentsIntent,
    RetrieveArtefactIntent,
    ReviewExperimentIntent,
    UnrecognisedIntent,
)
from src.orchestration.router.routing_schema import WorkflowResult

# ---------------------------------------------------------------------------
# Internal dispatch helper
# ---------------------------------------------------------------------------


def _dispatch(
    intent: Intent,
    fn_name: str,
    fn,
    *args,
    **kwargs,
) -> WorkflowResult:
    t0 = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        return WorkflowResult(
            intent=intent,
            api_function=fn_name,
            result=result,
            elapsed_seconds=time.monotonic() - t0,
        )
    except Exception as exc:
        return WorkflowResult(
            intent=intent,
            api_function=fn_name,
            result=None,
            elapsed_seconds=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def route(
    intent: Intent,
    base: Path | str | None = None,
) -> WorkflowResult:
    """Dispatch an intent to the appropriate research API function.

    Args:
        intent: A typed intent produced by ``intents.parse()``.
        base: Optional path override for the experiments directory.
              Passed through to the underlying API call.

    Returns:
        A ``WorkflowResult`` containing the API return value and metadata.
        If the intent is ``UnrecognisedIntent``, returns an error result
        without making any API call.
    """
    if isinstance(intent, ReviewExperimentIntent):
        return _dispatch(
            intent,
            "run_llm_review",
            _api.run_llm_review,
            intent.experiment_name,
            provider=intent.provider,
            model=intent.model,
            base=base,
        )

    if isinstance(intent, CompareExperimentsIntent):
        return _dispatch(
            intent,
            "run_llm_comparative_review",
            _api.run_llm_comparative_review,
            intent.baseline,
            intent.candidate,
            provider=intent.provider,
            model=intent.model,
            base=base,
        )

    if isinstance(intent, GenerateIterationIntent):
        return _dispatch(
            intent,
            "generate_iteration_proposal",
            _api.generate_iteration_proposal,
            intent.experiment_name,
            provider=intent.provider,
            model=intent.model,
            base=base,
        )

    if isinstance(intent, BuildEvolutionChainIntent):
        return _dispatch(
            intent,
            "build_research_evolution_chain",
            _api.build_research_evolution_chain,
            intent.root_experiment,
            base=base,
        )

    if isinstance(intent, ListExperimentsIntent):
        if intent.tag or intent.strategy_pattern:
            return _dispatch(
                intent,
                "find_experiments",
                _api.find_experiments,
                tag=intent.tag,
                strategy_pattern=intent.strategy_pattern,
                base=base,
            )
        return _dispatch(
            intent,
            "list_all_experiments",
            _api.list_all_experiments,
            base=base,
        )

    if isinstance(intent, RankExperimentsIntent):
        return _dispatch(
            intent,
            "rank_experiments_by_sharpe",
            _api.rank_experiments_by_sharpe,
            base=base,
            descending=intent.descending,
        )

    if isinstance(intent, RetrieveArtefactIntent):
        return _dispatch(
            intent,
            "retrieve_artefact",
            _api.retrieve_artefact,
            intent.experiment_name,
            intent.key,
            base=base,
        )

    if isinstance(intent, BuildContextIntent):
        return _dispatch(
            intent,
            "build_llm_context",
            _api.build_llm_context,
            intent.experiment_name,
            base=base,
        )

    if isinstance(intent, GenerateDraftIntent):
        return _dispatch(
            intent,
            "generate_experiment_draft",
            _api.generate_experiment_draft,
            intent.experiment_name,
            provider=intent.provider,
            model=intent.model,
        )

    # UnrecognisedIntent — no API call
    assert isinstance(intent, UnrecognisedIntent)
    return WorkflowResult(
        intent=intent,
        api_function="",
        result=None,
        error=f"UnrecognisedIntent: {intent.reason or 'no matching workflow'}",
    )
