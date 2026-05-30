"""AI-agent integration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentTask:
    """Structured request for an AI research assistant."""

    name: str
    objective: str
    inputs: Mapping[str, Any] = field(default_factory=dict)


class ResearchAgent(Protocol):
    """Agent interface for research summaries, diagnostics, and reports."""

    def run(self, task: AgentTask) -> Any:
        """Execute an AI-assisted research task from structured inputs."""
        raise NotImplementedError
