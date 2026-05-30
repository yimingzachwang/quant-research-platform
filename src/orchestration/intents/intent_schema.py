"""Typed intent dataclasses for the orchestration natural-language interface.

Each dataclass represents one recognisable user intent.  All are frozen so
they can be used as dict keys and in sets.  The parser returns exactly one of
these; the router dispatches on the concrete type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_DEFAULT_PROVIDER = "anthropic"


# ---------------------------------------------------------------------------
# Concrete intent types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewExperimentIntent:
    """Run an LLM review of a single experiment."""

    experiment_name: str
    provider: str = _DEFAULT_PROVIDER
    model: str | None = None


@dataclass(frozen=True)
class CompareExperimentsIntent:
    """Compare two experiments through semantic LLM interpretation."""

    baseline: str
    candidate: str
    provider: str = _DEFAULT_PROVIDER
    model: str | None = None


@dataclass(frozen=True)
class GenerateIterationIntent:
    """Generate a research iteration proposal for an experiment."""

    experiment_name: str
    provider: str = _DEFAULT_PROVIDER
    model: str | None = None


@dataclass(frozen=True)
class BuildEvolutionChainIntent:
    """Build a research evolution chain rooted at one experiment."""

    root_experiment: str


@dataclass(frozen=True)
class ListExperimentsIntent:
    """List available experiments, optionally filtered by tag or strategy."""

    tag: str | None = None
    strategy_pattern: str | None = None


@dataclass(frozen=True)
class RankExperimentsIntent:
    """Rank experiments by Sharpe ratio."""

    descending: bool = True


@dataclass(frozen=True)
class RetrieveArtefactIntent:
    """Retrieve a named artefact from an experiment."""

    experiment_name: str
    key: str


@dataclass(frozen=True)
class BuildContextIntent:
    """Build a structured LLM context for an experiment."""

    experiment_name: str


@dataclass(frozen=True)
class GenerateDraftIntent:
    """Generate a config draft from the most recent iteration proposal."""

    experiment_name: str
    provider: str = _DEFAULT_PROVIDER
    model: str | None = None


@dataclass(frozen=True)
class UnrecognisedIntent:
    """Fallback when the text cannot be mapped to a known intent."""

    raw_text: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Union type alias
# ---------------------------------------------------------------------------

Intent = (
    ReviewExperimentIntent
    | CompareExperimentsIntent
    | GenerateIterationIntent
    | BuildEvolutionChainIntent
    | ListExperimentsIntent
    | RankExperimentsIntent
    | RetrieveArtefactIntent
    | BuildContextIntent
    | GenerateDraftIntent
    | UnrecognisedIntent
)
