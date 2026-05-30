"""Schema dataclasses for Phase 4 — Research Session Layer.

Three types:
  SessionStatus     — string constants for session lifecycle
  SessionEventType  — string constants for the seven event types
  SessionEvent      — one recorded research action
  ResearchSession   — full session record: goal, timeline, active state

The session is a structured lab notebook.  It records what the researcher
did and what the current state is.  It does not issue instructions or
execute anything.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants (plain classes — no enum, no state machine)
# ---------------------------------------------------------------------------


class SessionStatus:
    ACTIVE   = "active"
    PAUSED   = "paused"
    COMPLETE = "complete"


class SessionEventType:
    REVIEW_GENERATED             = "REVIEW_GENERATED"
    ITERATION_PROPOSAL_GENERATED = "ITERATION_PROPOSAL_GENERATED"
    DRAFT_GENERATED              = "DRAFT_GENERATED"
    DRAFT_VALIDATED              = "DRAFT_VALIDATED"
    DRAFT_APPROVED               = "DRAFT_APPROVED"
    YAML_RENDERED                = "YAML_RENDERED"
    EXPERIMENT_LINKED            = "EXPERIMENT_LINKED"


# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------


@dataclass
class SessionEvent:
    """One recorded research action within a ResearchSession.

    event_type:      One of the SessionEventType constants.
    experiment_name: The experiment this action was performed on.
    data:            Type-specific payload (see SessionEventType docstring).
                     Keys per type:
                       REVIEW_GENERATED:             provider
                       ITERATION_PROPOSAL_GENERATED: context_hash, research_focus
                       DRAFT_GENERATED:              draft_id, draft_hash, proposed_name
                       DRAFT_VALIDATED:              draft_id, is_valid, error_count
                       DRAFT_APPROVED:               draft_id, draft_hash
                       YAML_RENDERED:                draft_id, config_path
                       EXPERIMENT_LINKED:            new_experiment
    """

    event_id:        str
    event_type:      str
    timestamp:       str
    experiment_name: str
    data:            dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ResearchSession:
    """Lightweight research session record.

    Holds a research goal, a timeline of session events, and the current
    active state (experiment, draft).  All logic lives in session_manager.py.
    This dataclass is data-only.
    """

    session_id:        str
    research_goal:     str
    root_experiment:   str
    active_experiment: str
    status:            str   # SessionStatus constant
    created_at:        str   # ISO 8601
    updated_at:        str   # ISO 8601
    events:            list[SessionEvent] = field(default_factory=list)
    active_draft_id:   str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
