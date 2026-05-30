"""Report generation interfaces, placeholders, and Phase D2 artefact reporting."""

# Phase D2: artefact-driven static report generation
from src.reporting.report_builder import (
    ExperimentArtefacts,
    ReportPaths,
    generate_experiment_report,
    load_experiment_artefacts,
)

# Report governance spec + canonical presets
from src.reporting.report_spec import (
    AUDIT_REPORT,
    CANONICAL_SHOWCASE,
    COMPACT_REPORT,
    DIAGNOSTICS_REPORT,
    FULL_DEMO_REPORT,
    STANDARD_REPORT,
    ResearchReportSpec,
)

# Legacy scaffold (preserved for backward compatibility)
from src.reporting.interfaces import Report, ReportGenerator
from src.reporting.placeholders import MarkdownReportGenerator

__all__ = [
    # Phase D2
    "ExperimentArtefacts",
    "ReportPaths",
    "generate_experiment_report",
    "load_experiment_artefacts",
    # Report spec
    "ResearchReportSpec",
    "FULL_DEMO_REPORT",
    "COMPACT_REPORT",
    "STANDARD_REPORT",
    "CANONICAL_SHOWCASE",
    "DIAGNOSTICS_REPORT",
    "AUDIT_REPORT",
    # Legacy
    "MarkdownReportGenerator",
    "Report",
    "ReportGenerator",
]
