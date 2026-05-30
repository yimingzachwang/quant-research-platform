"""Config synthesis layer for Phase 3 — Draft Generation, Validation, and Rendering.

Pipeline:
  generate_draft()    — IterationProposal + base config → ExperimentDraft (via LLM)
  validate_draft()    — ExperimentDraft → DraftValidationResult (no side-effects)
  approve_draft()     — ExperimentDraft → approved ExperimentDraft (state transition)
  render_to_yaml()    — approved ExperimentDraft → configs/experiments/{name}.yaml
"""

from src.orchestration.config_generation.draft_generator import generate_draft, load_draft
from src.orchestration.config_generation.draft_schema import (
    DraftChange,
    DraftValidationResult,
    ExperimentDraft,
    apply_changes,
    compute_draft_hash,
)
from src.orchestration.config_generation.draft_validator import (
    approve_draft,
    validate_draft,
)
from src.orchestration.config_generation.yaml_renderer import render_to_yaml

__all__ = [
    "DraftChange",
    "ExperimentDraft",
    "DraftValidationResult",
    "compute_draft_hash",
    "apply_changes",
    "generate_draft",
    "load_draft",
    "validate_draft",
    "approve_draft",
    "render_to_yaml",
]
