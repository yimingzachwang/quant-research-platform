"""Schema for structured LLM review outputs."""

from __future__ import annotations

REVIEW_VERSION = "1.0"

# Canonical section keys returned in LLMReviewOutput.sections
SECTION_PERFORMANCE = "performance_assessment"
SECTION_SIGNAL_QUALITY = "signal_quality"
SECTION_VALIDATION = "validation_robustness"
SECTION_FAILURE_MODES = "failure_mode_analysis"
SECTION_FEATURE_CONTRIBUTION = "feature_contribution_interpretation"
SECTION_RECOMMENDATIONS = "recommendations"

ALL_SECTIONS = [
    SECTION_PERFORMANCE,
    SECTION_SIGNAL_QUALITY,
    SECTION_VALIDATION,
    SECTION_FAILURE_MODES,
    SECTION_FEATURE_CONTRIBUTION,
    SECTION_RECOMMENDATIONS,
]

# Supported provider identifiers
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_STUB = "stub"
