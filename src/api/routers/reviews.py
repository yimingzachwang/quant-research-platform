"""Review and iteration proposal endpoints.

Wraps:
  run_llm_review()             + record_session_event(REVIEW_GENERATED)
  generate_iteration_proposal() + record_session_event(ITERATION_PROPOSAL_GENERATED)
  run_llm_comparative_review()
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import CompareRequest, ProposalRequest, ReviewRequest, _asdict
from src.orchestration.api import research_api as _api
from src.orchestration.session.session_schema import SessionEventType

router = APIRouter(prefix="/sessions", tags=["reviews"])


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
# POST /api/sessions/{session_id}/review
# ---------------------------------------------------------------------------


@router.post("/{session_id}/review")
def post_review(session_id: str, body: ReviewRequest) -> dict:
    session = _session_or_404(session_id)
    review = _api.run_llm_review(
        experiment_name=body.experiment_name,
        provider=body.provider,
        model=body.model,
    )
    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.REVIEW_GENERATED,
        experiment_name=body.experiment_name,
        data={"provider": review.provider, "model": review.model},
    )
    return {
        "review": _asdict(review),
        **_session_response(session),
    }


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/proposal
# ---------------------------------------------------------------------------


@router.post("/{session_id}/proposal")
def post_proposal(session_id: str, body: ProposalRequest) -> dict:
    session = _session_or_404(session_id)
    proposal = _api.generate_iteration_proposal(
        experiment_name=body.experiment_name,
        provider=body.provider,
        model=body.model,
    )
    session = _api.record_session_event(
        session=session,
        event_type=SessionEventType.ITERATION_PROPOSAL_GENERATED,
        experiment_name=body.experiment_name,
        data={
            "context_hash": proposal.context_hash,
            "research_focus": proposal.research_focus,
        },
    )
    return {
        "proposal": _asdict(proposal),
        **_session_response(session),
    }


# ---------------------------------------------------------------------------
# POST /api/sessions/{session_id}/compare
# ---------------------------------------------------------------------------


@router.post("/{session_id}/compare")
def post_compare(session_id: str, body: CompareRequest) -> dict:
    _session_or_404(session_id)
    comparison = _api.run_llm_comparative_review(
        baseline_experiment=body.baseline,
        candidate_experiment=body.candidate,
        provider=body.provider,
        model=body.model,
    )
    return {"comparison": _asdict(comparison)}
