"""Pydantic contracts for data requests."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataType(StrEnum):
    """Supported V1 dataset families."""

    OHLCV = "ohlcv"
    MACRO = "macro"


class DataSource(StrEnum):
    """Supported V1 external data sources."""

    YFINANCE = "yfinance"
    FRED = "fred"


class DataFrequency(StrEnum):
    """Supported V1 sampling frequencies."""

    DAILY = "1d"


class DataRequest(BaseModel):
    """Strongly typed request for one deterministic dataset.

    The request intentionally describes one symbol or macro series at a time.
    Multi-asset orchestration should be explicit in calling workflow code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    symbol: str = Field(min_length=1, description="ETF ticker or macro series id.")
    data_type: DataType
    source: DataSource
    start_date: date
    end_date: date
    frequency: DataFrequency = DataFrequency.DAILY

    @model_validator(mode="after")
    def validate_supported_request(self) -> DataRequest:
        """Reject unsupported V1 combinations and invalid date ranges."""

        if self.start_date > self.end_date:
            msg = "start_date must be on or before end_date"
            raise ValueError(msg)

        if self.frequency is not DataFrequency.DAILY:
            msg = "Data Agent V1 only supports daily frequency"
            raise ValueError(msg)

        supported_pairs = {
            (DataType.OHLCV, DataSource.YFINANCE),
            (DataType.MACRO, DataSource.FRED),
        }
        if (self.data_type, self.source) not in supported_pairs:
            msg = f"unsupported V1 data_type/source pair: {self.data_type}/{self.source}"
            raise ValueError(msg)

        return self

    @property
    def normalized_symbol(self) -> str:
        """Return deterministic identifier casing used in storage and registry."""

        return self.symbol.upper()

    @property
    def dataset_id(self) -> str:
        """Stable registry identifier for this request."""

        return (
            f"{self.data_type.value}_{self.source.value}_"
            f"{self.normalized_symbol}_{self.frequency.value}"
        )
