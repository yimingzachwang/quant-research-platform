"""Static registry of known artefact types and their semantics.

This module answers "what does this artefact contain?" for any key in the
experiment output tree.  It is purely descriptive — no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtefactSpec:
    key: str
    filename: str
    group: str       # "core" | "diagnostics" | "research" | "plots" | "report"
    artefact_type: str  # "json" | "parquet" | "png" | "yaml" | "markdown" | "html"
    description: str
    required: bool = True


_CORE_ARTEFACTS: list[ArtefactSpec] = [
    ArtefactSpec("metadata", "metadata.json", "core", "json",
                 "Run provenance: name, strategy, parameters, timestamp."),
    ArtefactSpec("metrics", "metrics.json", "core", "json",
                 "Scalar performance metrics (Sharpe, return, drawdown, etc.)."),
    ArtefactSpec("config", "config.json", "core", "json",
                 "ExperimentSpec serialised as JSON.", required=False),
    ArtefactSpec("ml_provenance", "ml_provenance.json", "core", "json",
                 "ML model provenance: feature set, split dates, hyperparams.", required=False),
    ArtefactSpec("equity_curve", "equity_curve.parquet", "core", "parquet",
                 "Daily portfolio equity curve starting at 1.0."),
    ArtefactSpec("returns", "returns.parquet", "core", "parquet",
                 "Daily net portfolio return series."),
    ArtefactSpec("weights", "weights.parquet", "core", "parquet",
                 "Date × Asset applied portfolio weights."),
]

_DIAGNOSTICS_ARTEFACTS: list[ArtefactSpec] = [
    ArtefactSpec("backtest_diagnostics", "backtest_diagnostics.json", "diagnostics", "json",
                 "Rolling Sharpe/vol, drawdown windows, turnover statistics."),
    ArtefactSpec("ml_diagnostics", "ml_diagnostics.json", "diagnostics", "json",
                 "ML turnover, signal activity, weight periods by split."),
    ArtefactSpec("ml_model_diagnostics", "ml_model_diagnostics.json", "diagnostics", "json",
                 "IC series, coefficient stability, prediction strength, ranking geometry."),
    ArtefactSpec("split_metrics", "split_metrics.json", "diagnostics", "json",
                 "Walk-forward per-split and summary OOS metrics."),
    ArtefactSpec("universe_coverage", "universe_coverage.json", "diagnostics", "json",
                 "Monthly price coverage fraction per asset."),
    ArtefactSpec("wf_equity_curves", "wf_equity_curves.json", "diagnostics", "json",
                 "Per-split out-of-sample equity curves."),
]

_RESEARCH_ARTEFACTS: list[ArtefactSpec] = [
    ArtefactSpec("alignment_diagnostics", "alignment_diagnostics.json", "research", "json",
                 "Feature-return alignment: sample counts, alignment loss, panel stats."),
    ArtefactSpec("data_summary", "data_summary.json", "research", "json",
                 "Asset availability and data quality summary."),
    ArtefactSpec("feature_correlations", "feature_correlations.json", "research", "json",
                 "Pairwise Pearson correlation matrix of the feature space."),
    ArtefactSpec("feature_families", "feature_families.json", "research", "json",
                 "Feature-to-family grouping map."),
    ArtefactSpec("feature_registry", "feature_registry.json", "research", "json",
                 "Full feature definitions including construction parameters."),
    ArtefactSpec("feature_summary", "feature_summary.json", "research", "json",
                 "Per-feature descriptive statistics."),
    ArtefactSpec("signal_transitions", "signal_transitions.json", "research", "json",
                 "Rebalance frequency, transition counts, signal activity."),
]

ALL_ARTEFACTS: list[ArtefactSpec] = (
    _CORE_ARTEFACTS + _DIAGNOSTICS_ARTEFACTS + _RESEARCH_ARTEFACTS
)

_BY_KEY: dict[str, ArtefactSpec] = {spec.key: spec for spec in ALL_ARTEFACTS}


def get_spec(key: str) -> ArtefactSpec | None:
    return _BY_KEY.get(key)


def list_keys(group: str | None = None) -> list[str]:
    if group is None:
        return [s.key for s in ALL_ARTEFACTS]
    return [s.key for s in ALL_ARTEFACTS if s.group == group]
