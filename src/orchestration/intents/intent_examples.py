"""Canonical example inputs paired with their expected intent type.

Used by tests to verify rule-based parsing coverage and as documentation
for the kinds of natural language the parser is designed to handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Each entry: (raw_text, expected_intent_type_name, notes)
CANONICAL_EXAMPLES: list[tuple[str, str, str]] = [
    # ReviewExperimentIntent
    (
        "review canonical_ml_showcase",
        "ReviewExperimentIntent",
        "bare review verb + experiment name",
    ),
    (
        "analyze the results for canonical_ml_multi_asset",
        "ReviewExperimentIntent",
        "analyze synonym",
    ),
    (
        "give me an interpretation of canonical_ml_showcase",
        "ReviewExperimentIntent",
        "interpret synonym",
    ),
    (
        "assess canonical_ml_multi_asset performance",
        "ReviewExperimentIntent",
        "assess synonym",
    ),
    # CompareExperimentsIntent
    (
        "compare canonical_ml_showcase vs canonical_ml_multi_asset",
        "CompareExperimentsIntent",
        "vs keyword",
    ),
    (
        "compare canonical_ml_showcase against canonical_ml_multi_asset",
        "CompareExperimentsIntent",
        "against keyword",
    ),
    (
        "diff canonical_ml_showcase and canonical_ml_multi_asset",
        "CompareExperimentsIntent",
        "diff keyword",
    ),
    # GenerateIterationIntent
    (
        "generate an iteration proposal for canonical_ml_showcase",
        "GenerateIterationIntent",
        "explicit phrase",
    ),
    (
        "suggest improvements to canonical_ml_multi_asset",
        "GenerateIterationIntent",
        "suggest synonym",
    ),
    (
        "what should the next experiment after canonical_ml_showcase be",
        "GenerateIterationIntent",
        "next keyword",
    ),
    # BuildEvolutionChainIntent
    (
        "build evolution chain for canonical_ml_showcase",
        "BuildEvolutionChainIntent",
        "explicit phrase",
    ),
    (
        "show lineage of canonical_ml_showcase",
        "BuildEvolutionChainIntent",
        "lineage keyword",
    ),
    (
        "what is the research history rooted at canonical_ml_multi_asset",
        "BuildEvolutionChainIntent",
        "history keyword",
    ),
    # ListExperimentsIntent
    (
        "list all experiments",
        "ListExperimentsIntent",
        "bare list command",
    ),
    (
        "show me all available experiments",
        "ListExperimentsIntent",
        "show variant",
    ),
    (
        "what experiments do we have",
        "ListExperimentsIntent",
        "question form",
    ),
    # RankExperimentsIntent
    (
        "rank experiments by sharpe",
        "RankExperimentsIntent",
        "explicit rank + sharpe",
    ),
    (
        "which experiments perform best",
        "RankExperimentsIntent",
        "best keyword",
    ),
    (
        "show top experiments sorted by sharpe ratio",
        "RankExperimentsIntent",
        "top + sorted",
    ),
    # RetrieveArtefactIntent
    (
        "retrieve metrics from canonical_ml_showcase",
        "RetrieveArtefactIntent",
        "retrieve verb",
    ),
    (
        "get the diagnostics artefact from canonical_ml_multi_asset",
        "RetrieveArtefactIntent",
        "get verb + artefact keyword",
    ),
    # BuildContextIntent
    (
        "build llm context for canonical_ml_showcase",
        "BuildContextIntent",
        "explicit phrase",
    ),
    (
        "prepare context for canonical_ml_multi_asset",
        "BuildContextIntent",
        "prepare synonym",
    ),
    # GenerateDraftIntent
    (
        "generate draft for canonical_ml_showcase",
        "GenerateDraftIntent",
        "explicit draft keyword",
    ),
    (
        "create config draft for canonical_ml_multi_asset",
        "GenerateDraftIntent",
        "create config draft phrase",
    ),
    (
        "synthesize config for canonical_ml_showcase",
        "GenerateDraftIntent",
        "synthesize verb",
    ),
]
