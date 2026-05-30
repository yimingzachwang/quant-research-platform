"""Pydantic request schemas and serialization helpers for the API bridge.

Request models translate HTTP JSON payloads into existing Research API
arguments.  Response models are plain dicts — no parallel domain model.

The ``_asdict`` helper converts backend dataclasses to JSON-safe dicts using
existing ``.to_dict()`` methods where available, falling back to
``dataclasses.asdict()``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _asdict(obj: Any) -> Any:
    """Convert a backend dataclass to a JSON-serializable dict.

    Prefers .to_dict() when present, falls back to dataclasses.asdict().
    Returns the object unchanged if it is neither.
    """
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


# ---------------------------------------------------------------------------
# Session request schemas
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    root_experiment: str
    research_goal: str


class UpdateStatusRequest(BaseModel):
    status: str


class RecordEventRequest(BaseModel):
    event_type: str
    experiment_name: str
    data: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Review / proposal request schemas
# ---------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    experiment_name: str
    provider: str = "stub"
    model: str | None = None


class ProposalRequest(BaseModel):
    experiment_name: str
    provider: str = "stub"
    model: str | None = None


class CompareRequest(BaseModel):
    baseline: str
    candidate: str
    provider: str = "stub"
    model: str | None = None


# ---------------------------------------------------------------------------
# Draft request schemas
# ---------------------------------------------------------------------------


class DraftRequest(BaseModel):
    experiment_name: str
    provider: str = "stub"
    model: str | None = None


class DraftActionRequest(BaseModel):
    experiment_name: str
    draft_id: str


# ---------------------------------------------------------------------------
# Routing request schema
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    text: str
    session_id: str | None = None
    provider: str = "stub"
    model: str | None = None
