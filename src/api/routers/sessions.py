"""Session endpoints.

Wraps:
  list_research_sessions()
  create_research_session()
  load_research_session() + summarize_research_session()
  update_research_session_status()
  record_session_event()
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    CreateSessionRequest,
    RecordEventRequest,
    UpdateStatusRequest,
    _asdict,
)
from src.orchestration.api import research_api as _api
from src.orchestration.session.session_schema import SessionStatus

_VALID_STATUSES = {
    SessionStatus.ACTIVE,
    SessionStatus.PAUSED,
    SessionStatus.COMPLETE,
}

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _session_or_404(session_id: str) -> object:
    session = _api.load_research_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _session_response(session: object) -> dict:
    return {
        "session": _asdict(session),
        "summary": _api.summarize_research_session(session),
    }


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------


@router.get("")
def list_sessions() -> dict:
    return {"sessions": _api.list_research_sessions()}


# ---------------------------------------------------------------------------
# POST /api/sessions
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
def create_session(body: CreateSessionRequest) -> dict:
    session = _api.create_research_session(
        root_experiment=body.root_experiment,
        research_goal=body.research_goal,
    )
    return _session_response(session)


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}
# ---------------------------------------------------------------------------


@router.get("/{session_id}")
def get_session(session_id: str) -> dict:
    session = _session_or_404(session_id)
    return _session_response(session)


# ---------------------------------------------------------------------------
# PUT /api/sessions/{session_id}/status
# ---------------------------------------------------------------------------


@router.put("/{session_id}/status")
def update_status(session_id: str, body: UpdateStatusRequest) -> dict:
    if body.status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{body.status}'. Must be one of: {sorted(_VALID_STATUSES)}",
        )
    session = _session_or_404(session_id)
    session = _api.update_research_session_status(session, body.status)
    return _session_response(session)


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/events
# ---------------------------------------------------------------------------


@router.post("/{session_id}/events")
def record_event(session_id: str, body: RecordEventRequest) -> dict:
    session = _session_or_404(session_id)
    session = _api.record_session_event(
        session=session,
        event_type=body.event_type,
        experiment_name=body.experiment_name,
        data=body.data,
    )
    return _session_response(session)
