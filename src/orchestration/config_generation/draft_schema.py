"""Schema dataclasses for Phase 3 — Config Synthesis.

Three types mirror the existing orchestration schema pattern (api/schemas.py):
  DraftChange           — one proposed parameter delta
  ExperimentDraft       — full draft: base + proposed name + ordered deltas
  DraftValidationResult — result of running validate_draft()

Two helpers used by generator, validator, and renderer:
  compute_draft_hash    — deterministic SHA-256 of draft content
  apply_changes         — apply DraftChange list to a config dict (deep copy)
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------


@dataclass
class DraftChange:
    """One proposed parameter change within an ExperimentDraft.

    section: Top-level config key ("model", "labels", "signal", etc.)
    field:   Dotted sub-path ("params.alpha", "type", "parameters.train_months")
             or one of: "entries.add", "entries.remove" for feature list edits.
    current_value: Read from the base config by the generator — NEVER from LLM.
    proposed_value: From LLM structured JSON. Type must match field schema.
    rationale: Natural language from the IterationProposal.
    """

    section: str
    field: str
    current_value: Any
    proposed_value: Any
    rationale: str


@dataclass
class ExperimentDraft:
    """Typed, persisted intermediate representation between an IterationProposal
    and an executable YAML configuration.

    Not executable on its own.  render_to_yaml() requires approved=True.
    """

    draft_id: str
    draft_hash: str

    base_experiment: str
    source_proposal_hash: str

    proposed_name: str
    changes: list[DraftChange]

    generated_at: str

    approved: bool = False
    approved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class DraftValidationResult:
    """Result of validate_draft(). Not embedded in ExperimentDraft."""

    is_valid: bool
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_draft_hash(
    base_experiment: str,
    proposed_name: str,
    changes: list[DraftChange],
) -> str:
    """Deterministic 12-char SHA-256 of the draft's content fields."""
    payload = {
        "base_experiment": base_experiment,
        "proposed_name": proposed_name,
        "changes": [dataclasses.asdict(c) for c in changes],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]


def apply_changes(base_config: dict[str, Any], changes: list[DraftChange]) -> dict[str, Any]:
    """Apply a DraftChange list to a deep copy of base_config and return it.

    Feature additions append a feature dict to features.entries.
    Feature removals filter by name from features.entries.
    All other changes navigate the dotted field path within the section.
    Never mutates base_config.
    """
    out = copy.deepcopy(base_config)
    for change in changes:
        if change.section == "features":
            if change.field == "entries.add":
                out.setdefault("features", {}).setdefault("entries", []).append(
                    change.proposed_value
                )
            elif change.field == "entries.remove":
                entries = out.get("features", {}).get("entries", [])
                out["features"]["entries"] = [
                    e for e in entries if e.get("name") != change.proposed_value
                ]
        else:
            target = out.setdefault(change.section, {})
            parts = change.field.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = change.proposed_value
    return out
