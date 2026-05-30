"""Dataset metadata models used by the registry."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.data.contracts import DataFrequency, DataSource, DataType
from src.data.models.validation import ValidationStatus


class DatasetMetadata(BaseModel):
    """Registry entry for one stored dataset."""

    model_config = ConfigDict(use_enum_values=True)

    dataset_id: str
    symbol: str
    data_type: DataType
    source: DataSource
    frequency: DataFrequency
    start_date: date
    end_date: date
    row_count: int
    validation_status: ValidationStatus
    last_updated_at: datetime
    file_path: Path
