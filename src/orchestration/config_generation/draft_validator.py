"""Draft validation and approval.

validate_draft() delegates the authoritative config check to the existing
validate_ml_config().  No validation logic is duplicated here.

approve_draft() is a pure state transition — four lines, no subsystem.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

from src.experiments.config_io import load_config
from src.experiments.ml_config import validate_ml_config
from src.orchestration.config_generation.draft_schema import (
    DraftValidationResult,
    ExperimentDraft,
    apply_changes,
    compute_draft_hash,
)
from src.orchestration.registry.experiment_registry import list_all
from src.orchestration.utils.filesystem import experiment_config_path

# _VALID_CHANGE_PATHS must stay in sync with the valid change path vocabulary
# listed in draft_generator._LLM_SYSTEM.  Update both together when the
# schema changes.
# Bounded vocabulary of valid (section, field) pairs.
# The LLM output is checked against this whitelist before calling validate_ml_config().
_VALID_CHANGE_PATHS: frozenset[tuple[str, str]] = frozenset({
    ("model", "type"),
    ("model", "params.alpha"),
    ("model", "params.C"),
    ("model", "params.l1_ratio"),
    ("model", "params.max_iter"),
    ("labels", "type"),
    ("labels", "params.horizon"),
    ("signal", "type"),
    ("signal", "params.n"),
    ("signal", "params.n_long"),
    ("signal", "params.n_short"),
    ("signal", "params.threshold"),
    ("validation", "parameters.train_months"),
    ("validation", "parameters.test_months"),
    ("validation", "parameters.gap_days"),
    ("execution", "transaction_cost_bps"),
    ("portfolio_construction", "weighting.scheme"),
    ("portfolio_construction", "weighting.prediction_normalization"),
    ("portfolio_construction", "weighting.temperature"),
    ("features", "entries.add"),
    ("features", "entries.remove"),
})


def validate_draft(
    draft: ExperimentDraft,
    base: Path | str | None = None,
    configs_base: Path | str | None = None,
) -> DraftValidationResult:
    """Validate a draft against the existing validate_ml_config() schema.

    Steps:
      1. Check all (section, field) pairs against the whitelist.
      2. Check proposed_name is not already registered.
      3. Load base config; check version == "2".
      4. Apply changes to a deep copy of the base config.
      5. Delegate to validate_ml_config() — the authoritative validator.

    Returns a DraftValidationResult.  Never raises — all errors are captured.

    Args:
        draft:        The draft to validate.
        base:         Override for the experiments results directory (registry lookup).
        configs_base: Override for configs/experiments/ (base YAML loading).
    """
    errors: list[str] = []

    # 1. Whitelist check — fast path: reject unknown paths before any I/O
    for change in draft.changes:
        if (change.section, change.field) not in _VALID_CHANGE_PATHS:
            errors.append(
                f"Unknown change path: section={change.section!r}, field={change.field!r}. "
                f"See _VALID_CHANGE_PATHS for the supported vocabulary."
            )
    if errors:
        return DraftValidationResult(is_valid=False, errors=errors)

    # 2. Name collision
    existing = list_all(base)
    if draft.proposed_name in existing:
        errors.append(
            f"Experiment name {draft.proposed_name!r} already exists in the registry."
        )

    # 3. Load base config
    config_path = experiment_config_path(draft.base_experiment, configs_base)
    try:
        base_config = load_config(config_path)
    except FileNotFoundError:
        errors.append(
            f"Base config not found: {config_path}. "
            "Ensure the source experiment has a version-2 YAML config."
        )
        return DraftValidationResult(is_valid=False, errors=errors)

    if str(base_config.get("version", "1")) != "2":
        errors.append(
            f"Draft generation only supported for ML experiments (config version 2). "
            f"Base experiment {draft.base_experiment!r} has version "
            f"{base_config.get('version')!r}."
        )
        return DraftValidationResult(is_valid=False, errors=errors)

    # 4 & 5. Apply changes and delegate to validate_ml_config()
    merged = apply_changes(base_config, draft.changes)
    merged["name"] = draft.proposed_name

    try:
        validate_ml_config(merged)
    except ValueError as exc:
        errors.append(str(exc))

    return DraftValidationResult(is_valid=not errors, errors=errors)


def approve_draft(draft: ExperimentDraft) -> ExperimentDraft:
    """Return a new ExperimentDraft with approved=True and a recomputed hash.

    The hash is recomputed to reflect the final approved state — any researcher
    edits to proposed_name or changes are captured before approval is recorded.
    Re-validation before calling this function is the researcher's responsibility.
    """
    new_hash = compute_draft_hash(
        draft.base_experiment, draft.proposed_name, draft.changes
    )
    return dataclasses.replace(
        draft,
        draft_hash=new_hash,
        approved=True,
        approved_at=datetime.now(UTC).isoformat(),
    )


