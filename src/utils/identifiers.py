"""Identifier helpers for experiments and artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def new_experiment_id(prefix: str = "exp") -> str:
    """Create a sortable-ish experiment identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"
