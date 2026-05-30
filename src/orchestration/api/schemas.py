"""Typed dataclasses for the orchestration public API.

These are plain Python dataclasses — no Pydantic, no runtime validation
overhead.  They serve as explicit contracts between the retrieval layer,
context builder, and LLM interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentMetadata:
    """Core identity fields for one experiment run."""

    experiment_name: str
    strategy_name: str
    parameters: dict[str, Any]
    created_at: str
    tags: list[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Scalar performance metrics from metrics.json."""

    annualized_return: float | None = None
    annualized_volatility: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    calmar_ratio: float | None = None
    hit_rate: float | None = None
    raw: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Walk-forward validation summary from split_metrics.json."""

    n_splits: int = 0
    mean_oos_sharpe: float | None = None
    std_oos_sharpe: float | None = None
    hit_rate_positive_sharpe: float | None = None
    mean_oos_return: float | None = None
    worst_max_drawdown: float | None = None
    per_split: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MLDiagnosticSummary:
    """Key ML quality metrics from ml_model_diagnostics.json."""

    model_type: str | None = None
    mean_ic: float | None = None
    ic_t_stat: float | None = None
    directional_accuracy: float | None = None
    mean_feature_ic: float | None = None
    dominant_family: str | None = None
    n_family_transitions: int | None = None
    mean_hhi: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureMode:
    """A detected failure mode with its severity and evidence."""

    name: str
    severity: str  # "critical" | "warning" | "info"
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtefactMetadata:
    """Describes a single discovered artefact file."""

    key: str
    path: str
    artefact_type: str  # "json" | "parquet" | "png" | "yaml"
    group: str  # "core" | "diagnostics" | "research" | "plots" | "report"
    exists: bool = True


@dataclass
class PlotMetadata:
    """Metadata for a single plot from plot_index.json."""

    name: str
    group: str
    importance: str
    caption: str
    path: str | None = None


@dataclass
class LLMContext:
    """Structured context package sent to the LLM review layer.

    All fields are plain Python primitives — no DataFrames, no numpy arrays,
    no matplotlib Figures.  The LLM receives only what it needs to interpret
    results; never raw infrastructure state.
    """

    experiment_name: str
    strategy_name: str
    tags: list[str]
    created_at: str
    performance: dict[str, Any]
    validation: dict[str, Any]
    ml_diagnostics: dict[str, Any]
    failure_modes: list[dict[str, Any]]
    feature_summary: dict[str, Any]
    universe_summary: dict[str, Any]
    available_plots: list[dict[str, str]]
    report_sections: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMReviewOutput:
    """Structured output from one LLM review call."""

    experiment_name: str
    provider: str
    model: str
    prompt_template: str
    review_text: str
    sections: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    generated_at: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ExperimentSummary:
    """Lightweight summary for listings, comparisons, and index views."""

    experiment_name: str
    strategy_name: str
    created_at: str
    tags: list[str]
    sharpe_ratio: float | None
    annualized_return: float | None
    max_drawdown: float | None
    has_ml: bool = False
    has_validation: bool = False
    artefact_root: str = ""


@dataclass
class ExperimentLineage:
    """Lightweight lineage metadata for one experiment.

    Descriptive only — does not trigger orchestration.
    The researcher authors lineage records manually.
    """

    experiment_name: str
    parent_experiment: str | None

    created_at: str
    registered_at: str

    iteration_reason: str | None

    derived_from_iteration: bool
    derived_from_comparison: bool

    context_hash: str


@dataclass
class EvolutionStep:
    """Diagnostics-derived summary of one step in a research evolution chain."""

    experiment_name: str

    key_improvements: list[str]
    new_risks: list[str]
    persistent_failures: list[str]
    validation_changes: list[str]

    research_direction: str


@dataclass
class ResearchEvolutionChain:
    """Chronological research evolution chain rooted at one experiment.

    Exposes visible institutional research cognition — not autonomous optimization.
    The chain is metadata-driven, provenance-aware, and human-authored.
    """

    root_experiment: str
    experiments: list[str]
    generated_at: str
    evolution_summary: str
    steps: list[EvolutionStep]


@dataclass
class ComparativeReview:
    """Structured comparative review of two experiments.

    Advisory only — interprets diagnostic differences, does not rank strategies
    for deployment or prescribe configuration changes.
    """

    baseline_experiment: str
    candidate_experiment: str

    generated_at: str
    context_hash: str

    overall_assessment: str

    validation_changes: list[str]
    instability_changes: list[str]
    feature_behavior_changes: list[str]
    robustness_changes: list[str]
    failure_mode_changes: list[str]
    key_tradeoffs: list[str]

    research_progression_summary: str
    confidence: str

    provider: str = ""
    model: str = ""
    prompt_template: str = ""


@dataclass
class IterationProposal:
    """Structured research iteration proposal generated from diagnostic context.

    Advisory only — does not prescribe parameter changes or autonomous execution.
    The researcher remains the decision-maker for all experiment design.
    """

    experiment_name: str
    generated_at: str
    context_hash: str

    research_focus: str
    rationale: str

    supporting_evidence: list[str]
    suggested_experiments: list[str]
    instability_signals: list[str]
    validation_concerns: list[str]
    feature_risks: list[str]

    confidence: str

    provider: str = ""
    model: str = ""
    prompt_template: str = ""
