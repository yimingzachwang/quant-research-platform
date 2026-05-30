"""Report consistency validation (G-SYNC-5).

Detects publication inconsistencies between persisted artefacts and the
figures/tables actually rendered in the report.  Called by render_report()
at the end of section assembly.

Principle: no active research component may be silently omitted from the
canonical report.  Violations are collected and surfaced as a structured
warning list, not exceptions (report generation continues regardless).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
    from src.reporting.report_builder import ExperimentArtefacts


@dataclass
class ConsistencyWarning:
    """A single publication inconsistency detected during report validation."""

    severity: str          # "warning" | "info"
    category: str          # e.g. "feature_omission", "family_mismatch", "stale_figure"
    message: str


@dataclass
class ConsistencyReport:
    """Collected validation results for one report rendering."""

    warnings: list[ConsistencyWarning] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    @property
    def n_warnings(self) -> int:
        return len(self.warnings)

    def as_markdown(self) -> str:
        """Render warnings as a compact markdown block for inclusion in appendix."""
        if not self.warnings:
            return ""
        lines: list[str] = []
        lines.append("> **Publication Consistency Notes**")
        lines.append("> ")
        for w in self.warnings:
            icon = "⚠️" if w.severity == "warning" else "ℹ️"
            lines.append(f"> {icon} **[{w.category}]** {w.message}")
        return "\n".join(lines)


def validate_report_consistency(
    artefacts: "ExperimentArtefacts",
    fig_map: "dict[str, Path] | None",
    claimed: "set[str] | None" = None,
) -> ConsistencyReport:
    """Run all consistency checks and return a ConsistencyReport.

    Args:
        artefacts: Loaded experiment artefacts.
        fig_map:   Dict of figure_name → Path for all figures available
                   for this report.
        claimed:   Set of figure names already embedded in the report body.

    Returns:
        ConsistencyReport with any violations found.
    """
    report = ConsistencyReport()
    available_figs = set(fig_map.keys()) if fig_map else set()
    embedded = claimed or set()

    _check_feature_registry_completeness(artefacts, available_figs, embedded, report)
    _check_family_ic_coverage(artefacts, available_figs, report)
    _check_figure_coverage(artefacts, available_figs, embedded, report)
    _check_feature_ordering_consistency(artefacts, report)

    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_feature_registry_completeness(
    artefacts: "ExperimentArtefacts",
    available_figs: set[str],
    embedded: set[str],
    report: ConsistencyReport,
) -> None:
    """Verify that all active features surface in at least one figure."""
    fr = artefacts.feature_registry
    if not isinstance(fr, dict):
        return
    feat_entries = fr.get("features") or []
    if not feat_entries:
        return

    active_names = {e.get("name", "") for e in feat_entries}

    # Features should appear in correlation heatmap (if >=2 features exist)
    fc = artefacts.feature_correlations
    if isinstance(fc, dict) and len(active_names) >= 2:
        fc_features = set(fc.get("features") or [])
        missing = active_names - fc_features
        for name in sorted(missing):
            report.warnings.append(ConsistencyWarning(
                severity="warning",
                category="feature_omission",
                message=(
                    f"Active feature '{name}' is absent from the feature correlation "
                    "artefact — it may have been dropped during alignment."
                ),
            ))

    # Check that feature_summary covers all registry features
    fs = artefacts.feature_summary
    if isinstance(fs, dict):
        fs_features = set((fs.get("features") or {}).keys())
        missing_stats = active_names - fs_features
        for name in sorted(missing_stats):
            report.warnings.append(ConsistencyWarning(
                severity="info",
                category="feature_stats_gap",
                message=(
                    f"Active feature '{name}' has no entry in feature_summary.json. "
                    "Statistics may be incomplete."
                ),
            ))


def _check_family_ic_coverage(
    artefacts: "ExperimentArtefacts",
    available_figs: set[str],
    report: ConsistencyReport,
) -> None:
    """Verify that all active feature families are represented in IC diagnostics."""
    ff = artefacts.feature_families
    if not isinstance(ff, dict):
        return
    families = ff.get("families") or {}
    if not families:
        return

    # feature_family_ic figure should exist if there are multiple families
    if len(families) > 1 and "feature_family_ic" not in available_figs:
        report.warnings.append(ConsistencyWarning(
            severity="warning",
            category="family_ic_absent",
            message=(
                f"Experiment has {len(families)} active feature families but "
                "no 'feature_family_ic' figure was generated. Run with "
                "walk-forward validation to produce family IC diagnostics."
            ),
        ))

    # Feature IC heatmap should also exist
    mmd = artefacts.ml_model_diagnostics
    if isinstance(mmd, dict) and "feature_ic_by_split" in mmd:
        if "feature_ic_heatmap" not in available_figs:
            report.warnings.append(ConsistencyWarning(
                severity="warning",
                category="feature_ic_figure_absent",
                message=(
                    "feature_ic_by_split data is present in ml_model_diagnostics "
                    "but no 'feature_ic_heatmap' figure was generated."
                ),
            ))


def _check_figure_coverage(
    artefacts: "ExperimentArtefacts",
    available_figs: set[str],
    embedded: set[str],
    report: ConsistencyReport,
) -> None:
    """Warn if important figures exist but were never embedded in the report."""
    # Figures expected to be embedded inline; appendix-only figures excluded
    important_figs = {
        "feature_correlation_heatmap",
        "ml_feature_regimes",
        "feature_ic_heatmap",
        "feature_family_ic",
        "ml_coefficient_sign_heatmap",
        "walk_forward_timeline",
    }
    generated_but_unclaimed = (available_figs & important_figs) - embedded
    for fig_name in sorted(generated_but_unclaimed):
        report.warnings.append(ConsistencyWarning(
            severity="info",
            category="unclaimed_figure",
            message=(
                f"Figure '{fig_name}' was generated but not embedded in any "
                "report section. It will appear in the appendix only."
            ),
        ))


def _check_feature_ordering_consistency(
    artefacts: "ExperimentArtefacts",
    report: ConsistencyReport,
) -> None:
    """Verify that correlation matrix and registry have the same feature set."""
    fr = artefacts.feature_registry
    fc = artefacts.feature_correlations
    if not isinstance(fr, dict) or not isinstance(fc, dict):
        return

    registry_names = {e.get("name", "") for e in (fr.get("features") or [])}
    corr_names = set(fc.get("features") or [])

    if registry_names and corr_names and registry_names != corr_names:
        extra_in_corr = corr_names - registry_names
        missing_in_corr = registry_names - corr_names
        if extra_in_corr:
            report.warnings.append(ConsistencyWarning(
                severity="info",
                category="feature_set_divergence",
                message=(
                    f"Correlation matrix contains features not in registry: "
                    f"{sorted(extra_in_corr)}. Likely added dynamically."
                ),
            ))
        if missing_in_corr:
            report.warnings.append(ConsistencyWarning(
                severity="warning",
                category="feature_set_divergence",
                message=(
                    f"Registry features absent from correlation matrix: "
                    f"{sorted(missing_in_corr)}. These were dropped during alignment."
                ),
            ))
