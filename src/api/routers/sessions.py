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
    ExecuteApprovedConfigRequest,
    RecordEventRequest,
    UpdateStatusRequest,
    _asdict,
)
from src.orchestration.api import research_api as _api
from src.orchestration.session.session_schema import SessionEventType, SessionStatus

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


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/execute-approved-config
#
# Governed, human-controlled execution bridge.  Reaching this endpoint IS the
# researcher's explicit authorisation: it runs exactly one already-approved
# config through the existing engine (via the Research API), then analyses the
# generated artefacts.  No loop, no retry, no automatic re-run.  A dry run
# returns the planned action and records nothing.
# ---------------------------------------------------------------------------


@router.post("/{session_id}/execute-approved-config")
def execute_approved_config_endpoint(
    session_id: str, body: ExecuteApprovedConfigRequest
) -> dict:
    session = _session_or_404(session_id)
    active_experiment = getattr(session, "active_experiment", None)

    # Dry run: report the planned action only — execute nothing, record nothing.
    if body.dry_run:
        planned = _api.execute_and_review_approved_config(
            body.config_path,
            provider=body.provider,
            model=body.model,
            base_url=body.base_url,
            report=body.report,
            preset=body.preset,
            dry_run=True,
        )
        return {
            "execution": _asdict(planned.execution),
            "review": None,
            "session": _asdict(session),
            "summary": _api.summarize_research_session(session),
            "planned": True,
        }

    # Real execution — explicit authorisation recorded before running.
    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.EXECUTION_REQUESTED,
        experiment_name=active_experiment,
        data={
            "config_path": body.config_path,
            "preset": body.preset,
            "report": body.report,
        },
    )

    result = _api.execute_and_review_approved_config(
        body.config_path,
        provider=body.provider,
        model=body.model,
        base_url=body.base_url,
        report=body.report,
        preset=body.preset,
        dry_run=False,
    )
    execution = result.execution

    if not execution.success:
        # No post-run review on a failed execution.
        return {
            "execution": _asdict(execution),
            "review": None,
            "session": _asdict(session),
            "summary": _api.summarize_research_session(session),
            "error": execution.error,
        }

    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.EXECUTION_COMPLETED,
        experiment_name=execution.experiment_name or active_experiment,
        data={
            "config_path": execution.config_path,
            "experiment_name": execution.experiment_name,
            "artefact_root": execution.artefact_root,
            "report_path": execution.report_path,
        },
    )

    if result.review is not None:
        session = _api.record_session_event(
            session=session,
            event_type=SessionEventType.POST_RUN_REVIEW_GENERATED,
            experiment_name=execution.experiment_name or active_experiment,
            data={
                "experiment_name": execution.experiment_name,
                "context_hash": result.context_hash,
                "provider": body.provider,
            },
        )

    return {
        "execution": _asdict(execution),
        "review": _asdict(result.review),
        "session": _asdict(session),
        "summary": _api.summarize_research_session(session),
    }
