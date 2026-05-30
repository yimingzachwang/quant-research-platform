"""Unit tests for orchestration.router: WorkflowResult schema and routing dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.orchestration.intents.intent_schema import (
    BuildContextIntent,
    BuildEvolutionChainIntent,
    CompareExperimentsIntent,
    GenerateIterationIntent,
    ListExperimentsIntent,
    RankExperimentsIntent,
    RetrieveArtefactIntent,
    ReviewExperimentIntent,
    UnrecognisedIntent,
)
from src.orchestration.router.routing_schema import WorkflowResult
from src.orchestration.router.workflow_router import route

# ---------------------------------------------------------------------------
# WorkflowResult schema
# ---------------------------------------------------------------------------


def test_workflow_result_success_flag():
    intent = ListExperimentsIntent()
    r = WorkflowResult(intent=intent, api_function="list_all_experiments", result=["exp_a"])
    assert r.success is True


def test_workflow_result_error_flag():
    intent = ListExperimentsIntent()
    r = WorkflowResult(
        intent=intent,
        api_function="list_all_experiments",
        result=None,
        error="ValueError: something went wrong",
    )
    assert r.success is False


def test_workflow_result_frozen():
    intent = ListExperimentsIntent()
    r = WorkflowResult(intent=intent, api_function="fn", result=None)
    with pytest.raises((TypeError, AttributeError)):
        r.result = "mutated"  # type: ignore[misc]


def test_workflow_result_stores_intent():
    intent = RankExperimentsIntent()
    r = WorkflowResult(intent=intent, api_function="rank_experiments_by_sharpe", result=[])
    assert r.intent is intent


def test_workflow_result_elapsed_defaults_to_zero():
    intent = ListExperimentsIntent()
    r = WorkflowResult(intent=intent, api_function="fn", result=None)
    assert r.elapsed_seconds == 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_MODULE = "src.orchestration.router.workflow_router._api"


def _mock_route(intent, mock_fn_name: str, return_value, base=None):
    """Call route() with a patched API function and return the WorkflowResult."""
    with patch(f"{_API_MODULE}.{mock_fn_name}", return_value=return_value) as m:
        result = route(intent, base=base)
    return result, m


# ---------------------------------------------------------------------------
# ReviewExperimentIntent
# ---------------------------------------------------------------------------


def test_route_review_calls_run_llm_review():
    intent = ReviewExperimentIntent(experiment_name="exp_a", provider="stub")
    sentinel = MagicMock(name="LLMReviewOutput")
    result, mock_fn = _mock_route(intent, "run_llm_review", sentinel)

    assert result.success
    assert result.api_function == "run_llm_review"
    assert result.result is sentinel
    mock_fn.assert_called_once_with("exp_a", provider="stub", model=None, base=None)


def test_route_review_propagates_model_override():
    intent = ReviewExperimentIntent(experiment_name="exp_a", provider="openai", model="gpt-4o")
    with patch(f"{_API_MODULE}.run_llm_review", return_value=None) as m:
        route(intent)
    m.assert_called_once_with("exp_a", provider="openai", model="gpt-4o", base=None)


# ---------------------------------------------------------------------------
# CompareExperimentsIntent
# ---------------------------------------------------------------------------


def test_route_compare_calls_comparative_review():
    intent = CompareExperimentsIntent(baseline="exp_a", candidate="exp_b", provider="stub")
    sentinel = MagicMock(name="ComparativeReview")
    result, mock_fn = _mock_route(intent, "run_llm_comparative_review", sentinel)

    assert result.success
    assert result.api_function == "run_llm_comparative_review"
    mock_fn.assert_called_once_with("exp_a", "exp_b", provider="stub", model=None, base=None)


# ---------------------------------------------------------------------------
# GenerateIterationIntent
# ---------------------------------------------------------------------------


def test_route_iterate_calls_generate_iteration_proposal():
    intent = GenerateIterationIntent(experiment_name="exp_a", provider="stub")
    sentinel = MagicMock(name="IterationProposal")
    result, mock_fn = _mock_route(intent, "generate_iteration_proposal", sentinel)

    assert result.success
    assert result.api_function == "generate_iteration_proposal"
    mock_fn.assert_called_once_with("exp_a", provider="stub", model=None, base=None)


# ---------------------------------------------------------------------------
# BuildEvolutionChainIntent
# ---------------------------------------------------------------------------


def test_route_evolution_calls_build_research_evolution_chain():
    intent = BuildEvolutionChainIntent(root_experiment="exp_a")
    sentinel = MagicMock(name="ResearchEvolutionChain")
    result, mock_fn = _mock_route(intent, "build_research_evolution_chain", sentinel)

    assert result.success
    assert result.api_function == "build_research_evolution_chain"
    mock_fn.assert_called_once_with("exp_a", base=None)


# ---------------------------------------------------------------------------
# ListExperimentsIntent
# ---------------------------------------------------------------------------


def test_route_list_no_filter_calls_list_all():
    intent = ListExperimentsIntent()
    result, mock_fn = _mock_route(intent, "list_all_experiments", ["exp_a", "exp_b"])

    assert result.success
    assert result.api_function == "list_all_experiments"
    mock_fn.assert_called_once_with(base=None)


def test_route_list_with_tag_calls_find_experiments():
    intent = ListExperimentsIntent(tag="ml")
    with patch(f"{_API_MODULE}.find_experiments", return_value=["exp_a"]) as m:
        result = route(intent)
    assert result.api_function == "find_experiments"
    m.assert_called_once_with(tag="ml", strategy_pattern=None, base=None)


def test_route_list_with_strategy_calls_find_experiments():
    intent = ListExperimentsIntent(strategy_pattern="ridge")
    with patch(f"{_API_MODULE}.find_experiments", return_value=[]) as m:
        result = route(intent)
    assert result.api_function == "find_experiments"
    m.assert_called_once_with(tag=None, strategy_pattern="ridge", base=None)


# ---------------------------------------------------------------------------
# RankExperimentsIntent
# ---------------------------------------------------------------------------


def test_route_rank_descending():
    intent = RankExperimentsIntent(descending=True)
    result, mock_fn = _mock_route(intent, "rank_experiments_by_sharpe", [])
    assert result.api_function == "rank_experiments_by_sharpe"
    mock_fn.assert_called_once_with(base=None, descending=True)


def test_route_rank_ascending():
    intent = RankExperimentsIntent(descending=False)
    with patch(f"{_API_MODULE}.rank_experiments_by_sharpe", return_value=[]) as m:
        route(intent)
    m.assert_called_once_with(base=None, descending=False)


# ---------------------------------------------------------------------------
# RetrieveArtefactIntent
# ---------------------------------------------------------------------------


def test_route_retrieve_artefact():
    intent = RetrieveArtefactIntent(experiment_name="exp_a", key="metrics")
    sentinel = {"sharpe": 1.23}
    result, mock_fn = _mock_route(intent, "retrieve_artefact", sentinel)

    assert result.success
    assert result.result is sentinel
    mock_fn.assert_called_once_with("exp_a", "metrics", base=None)


# ---------------------------------------------------------------------------
# BuildContextIntent
# ---------------------------------------------------------------------------


def test_route_build_context():
    intent = BuildContextIntent(experiment_name="exp_a")
    sentinel = MagicMock(name="LLMContext")
    result, mock_fn = _mock_route(intent, "build_llm_context", sentinel)

    assert result.success
    assert result.api_function == "build_llm_context"
    mock_fn.assert_called_once_with("exp_a", base=None)


# ---------------------------------------------------------------------------
# UnrecognisedIntent
# ---------------------------------------------------------------------------


def test_route_unrecognised_returns_error_no_api_call():
    intent = UnrecognisedIntent(raw_text="frobnicate", reason="no match")
    result = route(intent)

    assert not result.success
    assert result.api_function == ""
    assert result.result is None
    assert "UnrecognisedIntent" in result.error


def test_route_unrecognised_empty_reason():
    intent = UnrecognisedIntent(raw_text="???")
    result = route(intent)
    assert not result.success


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_route_api_exception_captured_in_result():
    intent = ListExperimentsIntent()
    with patch(f"{_API_MODULE}.list_all_experiments", side_effect=FileNotFoundError("no dir")):
        result = route(intent)

    assert not result.success
    assert "FileNotFoundError" in result.error
    assert "no dir" in result.error
    assert result.result is None


def test_route_elapsed_seconds_populated():
    intent = ListExperimentsIntent()
    with patch(f"{_API_MODULE}.list_all_experiments", return_value=[]):
        result = route(intent)
    assert result.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# base path propagation
# ---------------------------------------------------------------------------


def test_route_passes_base_path_through():
    intent = ReviewExperimentIntent(experiment_name="exp_a", provider="stub")
    with patch(f"{_API_MODULE}.run_llm_review", return_value=None) as m:
        route(intent, base="/some/custom/path")
    m.assert_called_once_with("exp_a", provider="stub", model=None, base="/some/custom/path")
