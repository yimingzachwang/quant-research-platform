"""Output schema for the workflow router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.orchestration.intents.intent_schema import Intent


@dataclass(frozen=True)
class WorkflowResult:
    """The result of routing an intent through the research API.

    Attributes:
        intent: The parsed intent that triggered this workflow.
        api_function: Name of the research_api function that was called.
        result: The raw return value from the API function.
        elapsed_seconds: Wall-clock seconds spent in the API call.
        error: Non-empty string if the API call raised an exception.
    """

    intent: Intent
    api_function: str
    result: Any
    elapsed_seconds: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error
