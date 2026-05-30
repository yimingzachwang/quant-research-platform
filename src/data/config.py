"""Typed configuration loading for Data Agent V1 ingestion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core import DateRange, Universe
from src.data.contracts import DataFrequency, DataRequest, DataSource, DataType


class DataStorageConfig(BaseModel):
    """Configured storage locations for Data Agent V1."""

    model_config = ConfigDict(extra="forbid")

    raw: Path
    processed: Path
    features: Path | None = None
    metadata: Path
    registry: Path


class DataValidationConfig(BaseModel):
    """Configured dataset validation policy."""

    model_config = ConfigDict(extra="forbid")

    max_nan_ratio: float = Field(ge=0, le=1)
    reject_empty: bool = True
    reject_duplicate_timestamps: bool = True
    reject_non_monotonic_timestamps: bool = True
    warn_missing_daily_timestamps: bool = True


class DataAgentV1Config(BaseModel):
    """Canonical Data Agent V1 ingestion configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    schema_version: str = Field(min_length=1)
    description: str | None = None
    supported_data_types: tuple[DataType, ...]
    supported_sources: dict[DataType, tuple[DataSource, ...]]
    frequency: DataFrequency
    storage: DataStorageConfig
    validation: DataValidationConfig
    canonical_schemas: dict[DataType, tuple[str, ...]] = Field(default_factory=dict)
    examples: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("supported_sources", mode="before")
    @classmethod
    def _coerce_supported_source_keys(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {DataType(str(key)): sources for key, sources in value.items()}

    @field_validator("canonical_schemas", mode="before")
    @classmethod
    def _coerce_schema_keys(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {DataType(str(key)): columns for key, columns in value.items()}

    @model_validator(mode="after")
    def validate_supported_sources(self) -> DataAgentV1Config:
        """Ensure each configured data type has at least one supported source."""

        missing = set(self.supported_data_types) - set(self.supported_sources)
        if missing:
            msg = f"supported_sources missing entries for: {sorted(item.value for item in missing)}"
            raise ValueError(msg)
        return self

    def validate_request_options(
        self,
        data_type: DataType,
        source: DataSource,
        frequency: DataFrequency,
    ) -> None:
        """Validate request options against the configured V1 contract."""

        if data_type not in self.supported_data_types:
            msg = f"unsupported configured data_type: {data_type.value}"
            raise ValueError(msg)
        if source not in self.supported_sources[data_type]:
            msg = f"unsupported configured source for {data_type.value}: {source.value}"
            raise ValueError(msg)
        if frequency is not self.frequency:
            msg = f"unsupported configured frequency: {frequency.value}"
            raise ValueError(msg)


class DateRangeConfig(BaseModel):
    """Date range configured for a dataset profile."""

    model_config = ConfigDict(extra="forbid")

    start: date
    end: date

    @model_validator(mode="after")
    def validate_date_order(self) -> DateRangeConfig:
        """Reject inverted date ranges."""

        if self.start > self.end:
            msg = "date_range.start must be on or before date_range.end"
            raise ValueError(msg)
        return self


class DatasetProfileConfig(BaseModel):
    """Dataset-level profile that references the canonical ingestion config."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    schema_version: str = Field(min_length=1)
    ingestion_config: Path
    universe: Path | None = None
    date_range: DateRangeConfig | None = None
    data_type: DataType
    source: DataSource
    frequency: DataFrequency


def load_data_agent_v1_config(path: str | Path) -> DataAgentV1Config:
    """Load the canonical Data Agent V1 YAML config."""

    return DataAgentV1Config.model_validate(_load_yaml(path))


def load_dataset_profile_config(path: str | Path) -> DatasetProfileConfig:
    """Load a dataset profile YAML config."""

    return DatasetProfileConfig.model_validate(_load_yaml(path))


def expand_data_requests(
    *,
    universe: Universe,
    date_range: DateRange,
    data_type: DataType,
    source: DataSource,
    frequency: DataFrequency,
    config: DataAgentV1Config,
) -> tuple[DataRequest, ...]:
    """Build one validated DataRequest per universe symbol."""

    config.validate_request_options(
        data_type=data_type,
        source=source,
        frequency=frequency,
    )
    return tuple(
        DataRequest(
            symbol=symbol,
            data_type=data_type,
            source=source,
            start_date=date_range.start,
            end_date=date_range.end,
            frequency=frequency,
        )
        for symbol in universe.symbols
    )


def expand_profile_data_requests(
    *,
    profile: DatasetProfileConfig,
    config: DataAgentV1Config,
    universe: Universe,
    date_range: DateRange,
) -> tuple[DataRequest, ...]:
    """Build DataRequests from a dataset profile and canonical config."""

    return expand_data_requests(
        universe=universe,
        date_range=date_range,
        data_type=profile.data_type,
        source=profile.source,
        frequency=profile.frequency,
        config=config,
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return loaded


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def build_data_requests(
    *,
    universe: Universe | Sequence[str],
    start_date: date | str,
    end_date: date | str,
    data_type: DataType,
    source: DataSource,
    frequency: DataFrequency,
    config: DataAgentV1Config,
) -> tuple[DataRequest, ...]:
    """Convenience wrapper for request generation from primitive inputs."""

    resolved_universe = (
        universe
        if isinstance(universe, Universe)
        else Universe(name="ad_hoc", symbols=tuple(str(symbol) for symbol in universe))
    )
    return expand_data_requests(
        universe=resolved_universe,
        date_range=DateRange(start=_coerce_date(start_date), end=_coerce_date(end_date)),
        data_type=data_type,
        source=source,
        frequency=frequency,
        config=config,
    )
