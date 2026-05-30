"""Structured validation report models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ValidationStatus(StrEnum):
    """Validation outcome severity."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ValidationCheck(BaseModel):
    """Single validation check result."""

    name: str
    status: ValidationStatus
    message: str
    observed_value: float | int | str | None = None


class ValidationReport(BaseModel):
    """Structured validation output for a dataset."""

    dataset_id: str
    status: ValidationStatus
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checks: list[ValidationCheck]

    @property
    def is_failed(self) -> bool:
        """Return whether any severe check failed."""

        return self.status == ValidationStatus.FAILED
