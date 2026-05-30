"""Report generation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Report:
    """Generated report metadata and content."""

    title: str
    content: str
    artifacts: dict[str, Any] = field(default_factory=dict)


class ReportGenerator(Protocol):
    """Creates a reproducible report from experiment artifacts."""

    def generate(self, context: Any, results: Any) -> Report:
        """Return a report object."""
        raise NotImplementedError
