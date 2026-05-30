"""Placeholder report generation implementations."""

from __future__ import annotations

from typing import Any

from src.reporting.interfaces import Report


class MarkdownReportGenerator:
    """Simple report generator for scaffold validation."""

    def generate(self, context: Any, results: Any) -> Report:
        """Generate a minimal markdown report."""
        content = (
            f"# Research Report: {context.experiment_id}\n\n"
            "This is an architecture-only placeholder report.\n\n"
            "No alpha strategy has been implemented.\n"
        )
        return Report(title=context.experiment_id, content=content, artifacts={"results": results})
