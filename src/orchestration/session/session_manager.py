"""Session manager: create, load, record events, summarize.

This module owns all logic for the research session layer.  It uses
existing serialization utilities (dump_json / load_json) and path helpers
(session_json_path, list_session_ids) from utils/.

No Research API functions are called here.
No evolution-chain functions are called here.
No quant engine code is called here.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.orchestration.session.session_schema import (
    ResearchSession,
    SessionEvent,
    SessionEventType,
    SessionStatus,
)
from src.orchestration.utils.filesystem import (
    session_json_path,
)
from src.orchestration.utils.serialization import dump_json, load_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def create_session(
    root_experiment: str,
    research_goal: str,
    session_id: str | None = None,
    sessions_base: Path | str | None = None,
) -> ResearchSession:
    """Create a new ResearchSession and persist it to disk.

    Args:
        root_experiment: Starting experiment for this session.
        research_goal:   Human-authored description of the research question.
        session_id:      Optional explicit ID; generated if absent.
        sessions_base:   Override for results/research_sessions/.

    Returns:
        A new ResearchSession with status=ACTIVE, persisted to session.json.
    """
    sid = session_id or str(uuid.uuid4())
    now = _now()
    session = ResearchSession(
        session_id=sid,
        research_goal=research_goal,
        root_experiment=root_experiment,
        active_experiment=root_experiment,
        status=SessionStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    _persist(session, sessions_base)
    logger.info("Session %s created for experiment %r", sid, root_experiment)
    return session


def load_session(
    session_id: str,
    sessions_base: Path | str | None = None,
) -> ResearchSession | None:
    """Load a ResearchSession from disk.

    Returns None if the session file does not exist.  Never raises.
    """
    path = session_json_path(session_id, sessions_base)
    data = load_json(path)
    if data is None:
        return None
    return _deserialize(data)


def record_event(
    session: ResearchSession,
    event_type: str,
    experiment_name: str,
    data: dict[str, Any] | None = None,
    sessions_base: Path | str | None = None,
) -> ResearchSession:
    """Append a SessionEvent and persist the updated session.

    Side-effects on session state:
      DRAFT_GENERATED  → sets active_draft_id from data["draft_id"]
      YAML_RENDERED    → clears active_draft_id
      EXPERIMENT_LINKED → sets active_experiment from data["new_experiment"]

    Does not call any Research API or evolution-chain functions.

    Returns:
        The updated ResearchSession (same object, mutated in place and saved).
    """
    event = SessionEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=_now(),
        experiment_name=experiment_name,
        data=data or {},
    )
    session.events.append(event)
    session.updated_at = event.timestamp

    if event_type == SessionEventType.DRAFT_GENERATED:
        session.active_draft_id = (data or {}).get("draft_id")
    elif event_type == SessionEventType.YAML_RENDERED:
        session.active_draft_id = None
    elif event_type == SessionEventType.EXPERIMENT_LINKED:
        new_exp = (data or {}).get("new_experiment")
        if new_exp:
            session.active_experiment = new_exp

    _persist(session, sessions_base)
    return session


def update_session_status(
    session: ResearchSession,
    status: str,
    sessions_base: Path | str | None = None,
) -> ResearchSession:
    """Set session status and persist.

    No transition rules are enforced — the researcher is authoritative.

    Returns:
        The updated ResearchSession.
    """
    session.status = status
    session.updated_at = _now()
    _persist(session, sessions_base)
    return session


def summarize_session(session: ResearchSession) -> dict[str, Any]:
    """Project a ResearchSession into a flat summary dict.

    Pure computation over the in-memory event log.  No disk I/O.
    No external API calls.  Suitable for frontend consumption.

    Returns a dict with keys:
        session_id, research_goal, status, root_experiment,
        active_experiment, created_at, updated_at, event_count,
        latest_review, latest_proposal, active_draft,
        approved_config_path, experiments_visited
    """
    latest_review:    dict[str, Any] | None = None
    latest_proposal:  dict[str, Any] | None = None
    active_draft:     dict[str, Any] | None = None
    approved_config:  str | None = None
    visited_order:    list[str] = []
    visited_seen:     set[str] = set()

    for ev in session.events:
        # Track experiment visit order (deduplicated, insertion-ordered)
        exp = ev.experiment_name
        if exp not in visited_seen:
            visited_seen.add(exp)
            visited_order.append(exp)

        if ev.event_type == SessionEventType.REVIEW_GENERATED:
            latest_review = {"experiment_name": exp, **ev.data, "timestamp": ev.timestamp}

        elif ev.event_type == SessionEventType.ITERATION_PROPOSAL_GENERATED:
            latest_proposal = {"experiment_name": exp, **ev.data, "timestamp": ev.timestamp}

        elif ev.event_type == SessionEventType.DRAFT_GENERATED:
            active_draft = {
                "experiment_name": exp,
                **ev.data,
                "timestamp": ev.timestamp,
                "approved": False,
            }
            approved_config = None  # new draft supersedes previous render

        elif ev.event_type == SessionEventType.DRAFT_APPROVED:
            if active_draft is not None:
                active_draft = {**active_draft, "approved": True}

        elif ev.event_type == SessionEventType.YAML_RENDERED:
            approved_config = ev.data.get("config_path")
            active_draft = None  # draft is now rendered

    return {
        "session_id":           session.session_id,
        "research_goal":        session.research_goal,
        "status":               session.status,
        "root_experiment":      session.root_experiment,
        "active_experiment":    session.active_experiment,
        "created_at":           session.created_at,
        "updated_at":           session.updated_at,
        "event_count":          len(session.events),
        "latest_review":        latest_review,
        "latest_proposal":      latest_proposal,
        "active_draft":         active_draft,
        "approved_config_path": approved_config,
        "experiments_visited":  visited_order,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _persist(session: ResearchSession, sessions_base: Path | str | None) -> None:
    path = session_json_path(session.session_id, sessions_base)
    dump_json(session.to_dict(), path)


def _deserialize(data: dict[str, Any]) -> ResearchSession:
    events = [
        SessionEvent(**e)
        for e in data.get("events", [])
    ]
    return ResearchSession(
        session_id=data["session_id"],
        research_goal=data["research_goal"],
        root_experiment=data["root_experiment"],
        active_experiment=data["active_experiment"],
        status=data["status"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        events=events,
        active_draft_id=data.get("active_draft_id"),
    )
