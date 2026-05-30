"""Research Session Layer — Phase 4.

A lightweight, JSON-backed session record over existing orchestration actions.
The session answers: what is the research goal, what actions have been taken,
what experiment is active, and what is the current draft state?

Usage::

    from src.orchestration.session import (
        create_session, record_event, summarize_session, SessionEventType
    )

    session = create_session("canonical_ml_showcase", "Explore regularization sweep")
    session = record_event(
        session,
        SessionEventType.REVIEW_GENERATED,
        "canonical_ml_showcase",
        data={"provider": "anthropic"},
    )
    print(summarize_session(session))
"""

from src.orchestration.session.session_manager import (
    create_session,
    load_session,
    record_event,
    summarize_session,
    update_session_status,
)
from src.orchestration.session.session_schema import (
    ResearchSession,
    SessionEvent,
    SessionEventType,
    SessionStatus,
)

__all__ = [
    "ResearchSession",
    "SessionEvent",
    "SessionStatus",
    "SessionEventType",
    "create_session",
    "load_session",
    "record_event",
    "update_session_status",
    "summarize_session",
]
