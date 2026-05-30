"""Reporting governance specification.

ResearchReportSpec controls which sections are rendered in a generated report.
Pass an instance to generate_experiment_report() or render_report() to get
mode-specific output without touching any section implementation.

Design constraints:
  - Frozen dataclass — fully immutable, hashable, serialisable.
  - No dynamic dispatch, no plugin hooks, no template engines.
  - All section rendering remains inside markdown.py; this spec only governs
    which sections are called.
  - None (default) resolves to STANDARD_REPORT at the call site — the canonical
    default for all unspecified calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchReportSpec:
    """Governs which sections appear in a generated research report.

    All flags default to the most informative settings.  The presets below
    cover the common use cases; construct a custom instance for anything else.

    Flags:
        include_summary:             Bullet-list headline (period, universe, Sharpe).
        include_metadata:            Metadata table (experiment name, strategy, date).
        include_configuration:       Config snapshot (universe, params, cost, validation).
        include_metrics:             Full performance metrics table.
        include_ml_analysis:         ML Model Behaviour section (v2 only; auto-guarded).
        include_ml_provenance_detail: Thin Model & Features provenance section (v2 only).
                                     Redundant with Research Thesis; off by default.
                                     Enable for AUDIT_REPORT or full provenance trails.
        include_validation:          Walk-Forward Validation section.
        include_diagnostics:         Split metrics + ML diagnostics tables.
                                     Requires persisted diagnostics artefacts; renders
                                     a note if artefacts are absent rather than raising.
        include_figures:             Embedded figure images.
        include_provenance:          Hash provenance table (config hash, ML hash).
        include_thesis:              Research Thesis — hypothesis, economic rationale,
                                     risks, and scope.
        include_methodology:         Backtesting Methodology — timing convention,
                                     cost model, look-ahead prevention details.
        include_data_infrastructure: Data Infrastructure — universe coverage,
                                     alignment policy, NaN diagnostics.
        include_portfolio_process:   Portfolio Construction Process — signal-to-weight
                                     pipeline description and rebalance history.
                                     Semantically distinct from data_infrastructure;
                                     controls the process narrative, not data provenance.
        include_failure_analysis:    Failure Analysis — worst drawdown windows,
                                     regime behaviour, known failure modes.
        include_feature_engineering: Feature Engineering — feature registry,
                                     alignment diagnostics, feature statistics,
                                     multicollinearity summary. v2 (ML) only;
                                     section is omitted when no feature artefacts
                                     are present regardless of this flag.
        include_universe_section:    Universe Construction & Coverage — universe
                                     validity, asset availability, cross-asset
                                     correlation structure, missingness summary.
                                     Rendered when universe_coverage artefact is
                                     present; gracefully absent otherwise.
        include_allocation_research: Allocation Research — concentration dynamics
                                     (HHI, effective breadth, entropy), prediction
                                     dispersion, and confidence calibration. Panel
                                     mode only; section omitted for single-asset
                                     experiments or when allocation_diagnostics
                                     artefact is absent.
    """

    include_summary: bool = True
    include_metadata: bool = True
    include_configuration: bool = True
    include_metrics: bool = True
    include_ml_analysis: bool = True
    include_ml_provenance_detail: bool = False
    include_validation: bool = True
    include_diagnostics: bool = False
    include_figures: bool = True
    include_provenance: bool = True
    include_thesis: bool = True
    include_methodology: bool = True
    include_data_infrastructure: bool = True
    include_portfolio_process: bool = True
    include_failure_analysis: bool = True
    include_feature_engineering: bool = True
    include_universe_section: bool = True
    include_allocation_research: bool = True


# ---------------------------------------------------------------------------
# Canonical presets — publication identities
# ---------------------------------------------------------------------------

STANDARD_REPORT = ResearchReportSpec(
    include_provenance=False,
)
"""Full research narrative with validation and feature engineering.  No diagnostics
appendix and no provenance table.  Default for all unspecified calls.

Audience: peer review, colleague sharing, general research communication.
Verbosity: complete — all narrative sections, figures, validation, ML visibility.
"""

CANONICAL_SHOWCASE = ResearchReportSpec(
    include_diagnostics=True,
)
"""Complete institutional research dossier: all narrative sections, full diagnostics
appendix, full provenance.

Audience: archival, institutional review, frontend presentation.
Verbosity: maximum — adds Diagnostics Appendix and Provenance to STANDARD_REPORT.
"""

DIAGNOSTICS_REPORT = ResearchReportSpec(
    include_diagnostics=True,
    include_provenance=False,
)
"""Full research narrative with diagnostics appendix.  No provenance table.

Audience: depth analysis, colleague sharing of diagnostic-level results.
Verbosity: narrative + diagnostics — between STANDARD_REPORT and CANONICAL_SHOWCASE.
"""

COMPACT_REPORT = ResearchReportSpec(
    include_validation=False,
    include_diagnostics=False,
    include_figures=False,
    include_provenance=False,
    include_thesis=False,
    include_methodology=False,
    include_data_infrastructure=False,
    include_portfolio_process=False,
    include_failure_analysis=False,
    include_feature_engineering=False,
    include_universe_section=False,
    include_allocation_research=False,
)
"""Metrics + config only.  Suitable for quick summaries or email snapshots.

Audience: monitoring, dashboards, email digests.
Verbosity: minimal — summary, metrics, metadata, configuration only.
"""

AUDIT_REPORT = ResearchReportSpec(
    include_diagnostics=True,
    include_provenance=True,
    include_figures=True,
    include_thesis=False,
    include_ml_provenance_detail=True,
)
"""Complete research archive with diagnostics, provenance, and all figures.

Audience: compliance review, audit trails, institutional sign-off, full reproducibility.
Verbosity: maximum — all research artefacts preserved for complete inspection.
"""

# ---------------------------------------------------------------------------
# Legacy preset — retained for backward compatibility
# ---------------------------------------------------------------------------

FULL_DEMO_REPORT = ResearchReportSpec()
"""Legacy default preset.  All sections enabled except diagnostics appendix.

DEPRECATED: use STANDARD_REPORT (default) or CANONICAL_SHOWCASE instead.
Retained for backward compatibility only.  No longer the system default.
"""
