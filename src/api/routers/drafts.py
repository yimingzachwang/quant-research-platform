"""Draft lifecycle endpoints.

Wraps:
  generate_experiment_draft()  + record_session_event(DRAFT_GENERATED)
  load_experiment_draft()
  validate_experiment_draft()  + record_session_event(DRAFT_VALIDATED)
  approve_experiment_draft()   + record_session_event(DRAFT_APPROVED)
  render_draft_to_yaml()       + record_session_event(YAML_RENDERED)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import DraftActionRequest, DraftRequest, _asdict
from src.orchestration.api import research_api as _api
from src.orchestration.session.session_schema import SessionEventType

router = APIRouter(prefix="/sessions", tags=["drafts"])


def _session_or_404(session_id: str) -> object:
    session = _api.load_research_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _draft_or_404(experiment_name: str, draft_id: str) -> object:
    draft = _api.load_experiment_draft(experiment_name, draft_id)
    if draft is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft '{draft_id}' not found for experiment '{experiment_name}'",
        )
    return draft


def _session_response(session: object) -> dict:
    return {
        "session": _asdict(session),
        "summary": _api.summarize_research_session(session),
    }


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/draft
# ---------------------------------------------------------------------------


@router.post("/{session_id}/draft")
def post_draft(session_id: str, body: DraftRequest) -> dict:
    session = _session_or_404(session_id)
    draft = _api.generate_experiment_draft(
        experiment_name=body.experiment_name,
        provider=body.provider,
        model=body.model,
    )
    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.DRAFT_GENERATED,
        experiment_name=body.experiment_name,
        data={
            "draft_id": draft.draft_id,
            "draft_hash": draft.draft_hash,
            "proposed_name": draft.proposed_name,
        },
    )
    return {
        "draft": _asdict(draft),
        **_session_response(session),
    }


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/draft/validate
# ---------------------------------------------------------------------------


@router.post("/{session_id}/draft/validate")
def post_draft_validate(session_id: str, body: DraftActionRequest) -> dict:
    session = _session_or_404(session_id)
    draft = _draft_or_404(body.experiment_name, body.draft_id)
    result = _api.validate_experiment_draft(draft)
    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.DRAFT_VALIDATED,
        experiment_name=body.experiment_name,
        data={"draft_id": body.draft_id, "is_valid": result.is_valid},
    )
    return {
        "validation": _asdict(result),
        **_session_response(session),
    }


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/draft/approve
# ---------------------------------------------------------------------------


@router.post("/{session_id}/draft/approve")
def post_draft_approve(session_id: str, body: DraftActionRequest) -> dict:
    session = _session_or_404(session_id)
    draft = _draft_or_404(body.experiment_name, body.draft_id)
    approved = _api.approve_experiment_draft(draft)
    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.DRAFT_APPROVED,
        experiment_name=body.experiment_name,
        data={"draft_id": approved.draft_id, "draft_hash": approved.draft_hash},
    )
    return {
        "draft": _asdict(approved),
        **_session_response(session),
    }


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/draft/render
# ---------------------------------------------------------------------------


@router.post("/{session_id}/draft/render")
def post_draft_render(session_id: str, body: DraftActionRequest) -> dict:
    session = _session_or_404(session_id)
    draft = _draft_or_404(body.experiment_name, body.draft_id)

    if not draft.approved:
        raise HTTPException(
            status_code=400,
            detail=f"Draft '{body.draft_id}' must be approved before rendering to YAML",
        )

    yaml_str = _api.render_draft_to_yaml(draft)
    config_path = f"configs/experiments/{draft.proposed_name}.yaml"

    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.YAML_RENDERED,
        experiment_name=body.experiment_name,
        data={"draft_id": draft.draft_id, "config_path": config_path},
    )
    return {
        "yaml": yaml_str,
        "config_path": config_path,
        **_session_response(session),
    }
