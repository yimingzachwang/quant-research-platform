"""Incremental update orchestration for canonical datasets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.contracts import DataRequest
from src.data.downloaders import DataDownloader
from src.data.loaders.storage import DataStorage
from src.data.models import DatasetMetadata, ValidationReport
from src.data.registry import JsonDatasetRegistry
from src.data.transformers import MacroStandardizer, OHLCVStandardizer
from src.data.validators import DatasetValidator


class DatasetUpdateEngine:
    """Download, merge, validate, store, and register deterministic datasets."""

    def __init__(
        self,
        storage: DataStorage,
        registry: JsonDatasetRegistry,
        validator: DatasetValidator,
        downloaders: dict[str, DataDownloader],
    ) -> None:
        self.storage = storage
        self.registry = registry
        self.validator = validator
        self.downloaders = downloaders
        self.ohlcv_standardizer = OHLCVStandardizer()
        self.macro_standardizer = MacroStandardizer()

    def update(self, request: DataRequest) -> pd.DataFrame:
        """Incrementally update and return the stored canonical dataset."""

        existing = (
            self.storage.load_processed(request) if self.storage.has_processed(request) else None
        )
        request_to_download = self._missing_period_request(request, existing)
        if request_to_download is None:
            logger.bind(dataset_id=request.dataset_id).info("processed dataset is already current")
            return (
                self._slice_to_request(existing, request)
                if existing is not None
                else pd.DataFrame()
            )

        downloader = self.downloaders.get(request.source.value)
        if downloader is None:
            msg = f"no downloader registered for source {request.source.value}"
            raise ValueError(msg)

        raw = downloader.download(request_to_download)
        self.storage.write_raw(raw, request_to_download)
        standardized = self._standardize(raw, request_to_download)
        merged = self._merge(existing, standardized)
        report = self.validator.validate(merged, request)
        if report.is_failed:
            logger.bind(dataset_id=request.dataset_id, status=report.status.value).error(
                "validation failed; refusing to store processed dataset"
            )
            raise ValueError(f"validation failed for {request.dataset_id}")

        path = self.storage.write_processed(merged, request)
        self.storage.validation_report_path(request).write_text(
            report.model_dump_json(indent=2) + "\n"
        )
        self.registry.upsert(self._metadata_from_frame(merged, request, report, path))
        return self._slice_to_request(merged, request)

    def _missing_period_request(
        self,
        request: DataRequest,
        existing: pd.DataFrame | None,
    ) -> DataRequest | None:
        if existing is None or existing.empty:
            return request

        latest_timestamp = pd.to_datetime(existing["timestamp"], utc=True).max().date()
        if latest_timestamp >= request.end_date:
            return None
        next_date = latest_timestamp + timedelta(days=1)
        return request.model_copy(update={"start_date": next_date})

    def _standardize(self, frame: pd.DataFrame, request: DataRequest) -> pd.DataFrame:
        if request.data_type.value == "ohlcv":
            return self.ohlcv_standardizer.transform(frame, request)
        return self.macro_standardizer.transform(frame, request)

    def _merge(self, existing: pd.DataFrame | None, new_data: pd.DataFrame) -> pd.DataFrame:
        frames = [frame for frame in [existing, new_data] if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        return (
            merged.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _slice_to_request(self, frame: pd.DataFrame, request: DataRequest) -> pd.DataFrame:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.date
        return frame.loc[
            (timestamps >= request.start_date) & (timestamps <= request.end_date)
        ].reset_index(drop=True)

    def _metadata_from_frame(
        self,
        frame: pd.DataFrame,
        request: DataRequest,
        report: ValidationReport,
        path: Path,
    ) -> DatasetMetadata:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        return DatasetMetadata(
            dataset_id=request.dataset_id,
            symbol=request.normalized_symbol,
            data_type=request.data_type,
            source=request.source,
            frequency=request.frequency,
            start_date=timestamps.min().date(),
            end_date=timestamps.max().date(),
            row_count=len(frame),
            validation_status=report.status,
            last_updated_at=datetime.now(UTC),
            file_path=path,
        )
