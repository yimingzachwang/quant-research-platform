"""Natural-language routing endpoint.

Wraps:
  parse()
  route()

Does not mutate session state.  The caller is responsible for recording
events if the routed action should appear in a session timeline.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastapi import APIRouter

from src.api.schemas import RouteRequest
from src.orchestration.api import research_api as _api
from src.orchestration.intents.intent_parser import parse
from src.orchestration.router.workflow_router import route

router = APIRouter(tags=["routing"])


def _serialise_result(result: Any) -> Any:
    """Convert a WorkflowResult.result value to a JSON-safe type."""
    if result is None:
        return None
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    return str(result)


# ---------------------------------------------------------------------------
# POST /api/route
# ---------------------------------------------------------------------------


@router.post("/route")
def post_route(body: RouteRequest) -> dict:
    known = _api.list_all_experiments()
    intent = parse(body.text, known_experiments=known, provider=body.provider, model=body.model)
    workflow = route(intent)
    return {
        "intent_type": type(intent).__name__,
        "success": workflow.success,
        "result": _serialise_result(workflow.result),
        "error": workflow.error or None,
        "elapsed_seconds": workflow.elapsed_seconds,
    }
