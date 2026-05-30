"""Workflow routing layer — maps typed intents to research API calls."""

from src.orchestration.router.routing_schema import WorkflowResult
from src.orchestration.router.workflow_router import route

__all__ = ["route", "WorkflowResult"]
