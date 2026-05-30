"""Dataset manifests and reproducible request hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.contracts import DataRequest


class DatasetManifest(BaseModel):
    """Manifest entry for a materialized dataset."""

    dataset_id: str
    dataset_name: str
    symbol: str
    data_type: str
    source: str
    frequency: str
    start_date: date
    end_date: date
    schema_version: str
    storage_path: str
    row_count: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_hash: str


class DatasetQuery(BaseModel):
    """Typed exact-match query for registry-backed dataset resolution."""

    model_config = ConfigDict(extra="forbid")

    dataset_family: str = "ohlcv"
    symbol: str | None = None
    frequency: str | None = None
    source: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    schema_version: str | None = None
    required_columns: tuple[str, ...] = ()

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        """Normalize symbols to the same casing used by manifests."""

        return None if value is None else value.upper()

    @model_validator(mode="after")
    def require_identity_field(self) -> DatasetQuery:
        """Require at least one identity field so queries are inspectable."""

        if not any([self.dataset_family, self.symbol, self.dataset_id, self.dataset_name]):
            msg = "DatasetQuery must include at least one identity field"
            raise ValueError(msg)
        return self

    @property
    def data_type(self) -> str:
        """Compatibility alias for the current manifest field name."""

        return self.dataset_family

    def without_schema_version(self) -> DatasetQuery:
        """Return a copy that ignores schema compatibility filtering."""

        return self.model_copy(update={"schema_version": None})

    @classmethod
    def from_parts(
        cls,
        *,
        symbol: str | None = None,
        data_type: str | None = None,
        dataset_family: str | None = None,
        frequency: str | None = None,
        source: str | None = None,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        schema_version: str | None = None,
        required_columns: Sequence[str] = (),
    ) -> DatasetQuery:
        """Build a query while preserving the existing data_type keyword."""

        return cls(
            dataset_family=dataset_family or data_type or "ohlcv",
            symbol=symbol,
            frequency=frequency,
            source=source,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            schema_version=schema_version,
            required_columns=tuple(required_columns),
        )


def hash_request(request: DataRequest, schema_version: str = "v1") -> str:
    """Return a deterministic SHA256 hash for a data request and schema version."""

    payload = {
        "data_type": request.data_type.value,
        "end_date": request.end_date.isoformat(),
        "frequency": request.frequency.value,
        "schema_version": schema_version,
        "source": request.source.value,
        "start_date": request.start_date.isoformat(),
        "symbol": request.normalized_symbol,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
