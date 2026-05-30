"""Parquet-backed deterministic dataset storage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.contracts import DataRequest, DataType


class DataStorage:
    """Persist raw and processed datasets under the repository data hierarchy."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)
        self.data_root = self.project_root / "data"
        self.raw_root = self.data_root / "raw"
        self.processed_root = self.data_root / "processed"
        self.metadata_root = self.data_root / "external" / "metadata"
        self.cache_root = self.data_root / "external" / "cache"
        for path in [self.raw_root, self.processed_root, self.metadata_root, self.cache_root]:
            path.mkdir(parents=True, exist_ok=True)

    def processed_path(self, request: DataRequest) -> Path:
        """Return canonical processed parquet path for a request."""

        data_dir = "ohlcv" if request.data_type is DataType.OHLCV else "macro"
        return (
            self.processed_root
            / data_dir
            / request.normalized_symbol
            / f"{request.frequency.value}.parquet"
        )

    def validation_report_path(self, request: DataRequest) -> Path:
        """Return validation metadata path for a request."""

        return self.metadata_root / f"{request.dataset_id}_validation.json"

    def has_processed(self, request: DataRequest) -> bool:
        """Return whether processed data exists for the request."""

        return self.processed_path(request).exists()

    def load_processed(self, request: DataRequest) -> pd.DataFrame:
        """Load processed parquet data."""

        path = self.processed_path(request)
        logger.bind(dataset_id=request.dataset_id, path=str(path)).info("loading processed dataset")
        return pd.read_parquet(path, engine="pyarrow")

    def write_processed(self, frame: pd.DataFrame, request: DataRequest) -> Path:
        """Write processed parquet data using pyarrow."""

        path = self.processed_path(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, engine="pyarrow", index=False)
        logger.bind(dataset_id=request.dataset_id, path=str(path), rows=len(frame)).info(
            "stored processed dataset"
        )
        return path

    def write_raw(self, frame: pd.DataFrame, request: DataRequest) -> Path:
        """Write an immutable source extract for reproducibility."""

        path = (
            self.raw_root
            / request.source.value
            / request.data_type.value
            / request.normalized_symbol
            / (
                f"{request.start_date.isoformat()}_{request.end_date.isoformat()}_"
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.parquet"
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            msg = f"raw dataset already exists and will not be overwritten: {path}"
            raise FileExistsError(msg)
        frame.to_parquet(path, engine="pyarrow", index=False)
        logger.bind(dataset_id=request.dataset_id, path=str(path), rows=len(frame)).info(
            "stored raw source extract"
        )
        return path
