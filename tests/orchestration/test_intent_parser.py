"""Unit tests for orchestration.intents: rule-based parser and schema."""

from __future__ import annotations

import pytest

from src.orchestration.intents.intent_parser import parse, _rule_based_parse
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
from src.orchestration.intents.intent_examples import CANONICAL_EXAMPLES

KNOWN = ["canonical_ml_showcase", "canonical_ml_multi_asset"]


# ---------------------------------------------------------------------------
# Intent dataclass contracts
# ---------------------------------------------------------------------------


def test_intent_dataclasses_are_frozen():
    intent = ReviewExperimentIntent(experiment_name="exp_a")
    with pytest.raises((TypeError, AttributeError)):
        intent.experiment_name = "mutated"  # type: ignore[misc]


def test_intent_default_provider():
    r = ReviewExperimentIntent(experiment_name="x")
    assert r.provider == "anthropic"
    assert r.model is None


def test_unrecognised_intent_defaults():
    u = UnrecognisedIntent(raw_text="gibberish")
    assert u.reason == ""


def test_list_experiments_intent_defaults():
    i = ListExperimentsIntent()
    assert i.tag is None
    assert i.strategy_pattern is None


def test_rank_experiments_intent_default_descending():
    r = RankExperimentsIntent()
    assert r.descending is True


# ---------------------------------------------------------------------------
# Empty / whitespace input
# ---------------------------------------------------------------------------


def test_empty_string_returns_unrecognised():
    assert isinstance(parse(""), UnrecognisedIntent)


def test_whitespace_only_returns_unrecognised():
    assert isinstance(parse("   "), UnrecognisedIntent)


# ---------------------------------------------------------------------------
# ReviewExperimentIntent
# ---------------------------------------------------------------------------


def test_review_explicit_keyword():
    r = parse("review canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, ReviewExperimentIntent)
    assert r.experiment_name == "canonical_ml_showcase"


def test_analyze_synonym():
    r = parse("analyze canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, ReviewExperimentIntent)


def test_interpret_synonym():
    r = parse("give me an interpretation of canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, ReviewExperimentIntent)


def test_assess_synonym():
    r = parse("assess canonical_ml_multi_asset performance", known_experiments=KNOWN)
    assert isinstance(r, ReviewExperimentIntent)
    assert r.experiment_name == "canonical_ml_multi_asset"


def test_bare_experiment_name_defaults_to_review():
    r = parse("canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, ReviewExperimentIntent)


# ---------------------------------------------------------------------------
# CompareExperimentsIntent
# ---------------------------------------------------------------------------


def test_compare_vs_keyword():
    r = parse(
        "compare canonical_ml_showcase vs canonical_ml_multi_asset",
        known_experiments=KNOWN,
    )
    assert isinstance(r, CompareExperimentsIntent)
    assert r.baseline == "canonical_ml_showcase"
    assert r.candidate == "canonical_ml_multi_asset"


def test_compare_against_keyword():
    r = parse(
        "compare canonical_ml_showcase against canonical_ml_multi_asset",
        known_experiments=KNOWN,
    )
    assert isinstance(r, CompareExperimentsIntent)


def test_diff_keyword():
    r = parse(
        "diff canonical_ml_showcase and canonical_ml_multi_asset",
        known_experiments=KNOWN,
    )
    assert isinstance(r, CompareExperimentsIntent)


def test_compare_requires_two_experiments():
    # Only one experiment name — should not produce CompareExperimentsIntent
    r = parse("compare canonical_ml_showcase", known_experiments=KNOWN)
    assert not isinstance(r, CompareExperimentsIntent)


# ---------------------------------------------------------------------------
# GenerateIterationIntent
# ---------------------------------------------------------------------------


def test_generate_iteration_explicit():
    r = parse(
        "generate an iteration proposal for canonical_ml_showcase",
        known_experiments=KNOWN,
    )
    assert isinstance(r, GenerateIterationIntent)
    assert r.experiment_name == "canonical_ml_showcase"


def test_suggest_synonym():
    r = parse("suggest improvements to canonical_ml_multi_asset", known_experiments=KNOWN)
    assert isinstance(r, GenerateIterationIntent)


def test_next_experiment_keyword():
    r = parse(
        "what should the next experiment after canonical_ml_showcase be",
        known_experiments=KNOWN,
    )
    assert isinstance(r, GenerateIterationIntent)


# ---------------------------------------------------------------------------
# BuildEvolutionChainIntent
# ---------------------------------------------------------------------------


def test_build_evolution_chain_explicit():
    r = parse("build evolution chain for canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, BuildEvolutionChainIntent)
    assert r.root_experiment == "canonical_ml_showcase"


def test_lineage_keyword():
    r = parse("show lineage of canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, BuildEvolutionChainIntent)


def test_history_keyword():
    r = parse(
        "what is the research history rooted at canonical_ml_multi_asset",
        known_experiments=KNOWN,
    )
    assert isinstance(r, BuildEvolutionChainIntent)


# ---------------------------------------------------------------------------
# ListExperimentsIntent
# ---------------------------------------------------------------------------


def test_list_all_experiments():
    r = parse("list all experiments", known_experiments=KNOWN)
    assert isinstance(r, ListExperimentsIntent)
    assert r.tag is None
    assert r.strategy_pattern is None


def test_show_experiments():
    r = parse("show me all available experiments", known_experiments=KNOWN)
    assert isinstance(r, ListExperimentsIntent)


def test_what_experiments_question():
    r = parse("what experiments do we have", known_experiments=KNOWN)
    assert isinstance(r, ListExperimentsIntent)


def test_list_with_tag():
    r = parse("list experiments tag: ml", known_experiments=KNOWN)
    assert isinstance(r, ListExperimentsIntent)
    assert r.tag == "ml"


def test_list_with_strategy():
    r = parse("list experiments strategy: ridge", known_experiments=KNOWN)
    assert isinstance(r, ListExperimentsIntent)
    assert r.strategy_pattern == "ridge"


# ---------------------------------------------------------------------------
# RankExperimentsIntent
# ---------------------------------------------------------------------------


def test_rank_by_sharpe():
    r = parse("rank experiments by sharpe", known_experiments=KNOWN)
    assert isinstance(r, RankExperimentsIntent)
    assert r.descending is True


def test_best_keyword():
    r = parse("which experiments perform best", known_experiments=KNOWN)
    assert isinstance(r, RankExperimentsIntent)


def test_rank_ascending():
    r = parse("rank experiments by sharpe ascending", known_experiments=KNOWN)
    assert isinstance(r, RankExperimentsIntent)
    assert r.descending is False


def test_rank_worst():
    r = parse("show worst experiments by sharpe", known_experiments=KNOWN)
    assert isinstance(r, RankExperimentsIntent)
    assert r.descending is False


# ---------------------------------------------------------------------------
# RetrieveArtefactIntent
# ---------------------------------------------------------------------------


def test_retrieve_metrics():
    r = parse("retrieve metrics from canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, RetrieveArtefactIntent)
    assert r.experiment_name == "canonical_ml_showcase"
    assert r.key == "metrics"


def test_get_diagnostics():
    r = parse(
        "get the diagnostics artefact from canonical_ml_multi_asset",
        known_experiments=KNOWN,
    )
    assert isinstance(r, RetrieveArtefactIntent)
    assert r.key == "diagnostics"


# ---------------------------------------------------------------------------
# BuildContextIntent
# ---------------------------------------------------------------------------


def test_build_context_explicit():
    r = parse("build llm context for canonical_ml_showcase", known_experiments=KNOWN)
    assert isinstance(r, BuildContextIntent)
    assert r.experiment_name == "canonical_ml_showcase"


def test_prepare_context_synonym():
    r = parse("prepare context for canonical_ml_multi_asset", known_experiments=KNOWN)
    assert isinstance(r, BuildContextIntent)


# ---------------------------------------------------------------------------
# UnrecognisedIntent (stub provider, no LLM fallback)
# ---------------------------------------------------------------------------


def test_gibberish_returns_unrecognised():
    r = parse("xkcd frobnicate wibble", known_experiments=KNOWN)
    assert isinstance(r, UnrecognisedIntent)


def test_no_known_experiments_no_match():
    r = parse("review my_private_exp", known_experiments=[])
    assert isinstance(r, UnrecognisedIntent)


# ---------------------------------------------------------------------------
# CANONICAL_EXAMPLES coverage
# ---------------------------------------------------------------------------


def test_canonical_examples_parse_without_error():
    """Every CANONICAL_EXAMPLE must parse without raising an exception."""
    for text, expected_type, note in CANONICAL_EXAMPLES:
        result = parse(text, known_experiments=KNOWN)
        assert result is not None, f"parse() returned None for: {text!r} ({note})"


def test_canonical_examples_intent_types():
    """CANONICAL_EXAMPLES must classify to the expected intent type."""
    for text, expected_type, note in CANONICAL_EXAMPLES:
        result = parse(text, known_experiments=KNOWN)
        actual_type = type(result).__name__
        assert actual_type == expected_type, (
            f"Expected {expected_type} for {text!r} ({note}), got {actual_type}"
        )


# ---------------------------------------------------------------------------
# Provider wiring
# ---------------------------------------------------------------------------


def test_stub_provider_skips_llm_fallback():
    # Ambiguous text with no experiment name — rule-based fails, stub skips LLM
    r = parse("do something intelligent", known_experiments=KNOWN, provider="stub")
    assert isinstance(r, UnrecognisedIntent)
    assert "stub" in r.reason.lower() or "skipped" in r.reason.lower()
