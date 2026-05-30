"""Validation checks for canonical datasets."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.data.contracts import DataRequest, DataType
from src.data.models import ValidationCheck, ValidationReport, ValidationStatus


class DatasetValidator:
    """Validate canonical dataframes without silently repairing severe issues."""

    def __init__(self, max_nan_ratio: float = 0.05) -> None:
        if not 0 <= max_nan_ratio <= 1:
            msg = "max_nan_ratio must be between 0 and 1"
            raise ValueError(msg)
        self.max_nan_ratio = max_nan_ratio

    def validate(self, frame: pd.DataFrame, request: DataRequest) -> ValidationReport:
        """Validate a canonical dataframe and return a structured report."""

        checks: list[ValidationCheck] = []
        checks.append(self._check_empty(frame))

        required_columns = self._required_columns(request)
        checks.append(self._check_required_columns(frame, required_columns))

        if "timestamp" in frame.columns and not frame.empty:
            checks.append(self._check_duplicate_timestamps(frame))
            checks.append(self._check_monotonic_timestamps(frame))
            checks.append(self._check_missing_timestamps(frame, request))

        for column in required_columns - {
            "timestamp",
            "symbol",
            "series_id",
            "source",
            "frequency",
        }:
            if column in frame.columns:
                checks.append(self._check_nan_ratio(frame, column))

        status = self._aggregate_status(checks)
        report = ValidationReport(dataset_id=request.dataset_id, status=status, checks=checks)
        logger.bind(dataset_id=request.dataset_id, status=status.value).info("validated dataset")
        return report

    def _required_columns(self, request: DataRequest) -> set[str]:
        if request.data_type is DataType.OHLCV:
            return {
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "source",
                "frequency",
            }
        return {"timestamp", "value", "series_id", "source", "frequency"}

    def _check_empty(self, frame: pd.DataFrame) -> ValidationCheck:
        if frame.empty:
            return ValidationCheck(
                name="empty_dataset",
                status=ValidationStatus.FAILED,
                message="dataset contains no rows",
                observed_value=0,
            )
        return ValidationCheck(
            name="empty_dataset",
            status=ValidationStatus.PASSED,
            message="dataset contains rows",
            observed_value=len(frame),
        )

    def _check_required_columns(
        self, frame: pd.DataFrame, required_columns: set[str]
    ) -> ValidationCheck:
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            return ValidationCheck(
                name="required_columns",
                status=ValidationStatus.FAILED,
                message=f"dataset is missing required columns: {missing}",
                observed_value=",".join(missing),
            )
        return ValidationCheck(
            name="required_columns",
            status=ValidationStatus.PASSED,
            message="dataset contains required columns",
        )

    def _check_duplicate_timestamps(self, frame: pd.DataFrame) -> ValidationCheck:
        duplicate_count = int(frame["timestamp"].duplicated().sum())
        if duplicate_count:
            return ValidationCheck(
                name="duplicate_timestamps",
                status=ValidationStatus.FAILED,
                message="dataset contains duplicate timestamps",
                observed_value=duplicate_count,
            )
        return ValidationCheck(
            name="duplicate_timestamps",
            status=ValidationStatus.PASSED,
            message="timestamps are unique",
            observed_value=0,
        )

    def _check_monotonic_timestamps(self, frame: pd.DataFrame) -> ValidationCheck:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        if not timestamps.is_monotonic_increasing:
            return ValidationCheck(
                name="monotonic_ordering",
                status=ValidationStatus.FAILED,
                message="timestamps are not monotonically increasing",
            )
        return ValidationCheck(
            name="monotonic_ordering",
            status=ValidationStatus.PASSED,
            message="timestamps are monotonically increasing",
        )

    def _check_missing_timestamps(
        self, frame: pd.DataFrame, request: DataRequest
    ) -> ValidationCheck:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.normalize()
        if request.data_type is DataType.OHLCV:
            expected = pd.date_range(
                timestamps.min(),
                timestamps.max(),
                freq="B",
                tz="UTC",
            )
        else:
            expected = pd.date_range(
                timestamps.min(),
                timestamps.max(),
                freq="D",
                tz="UTC",
            )
        missing_count = int(len(expected.difference(pd.DatetimeIndex(timestamps))))
        if missing_count:
            return ValidationCheck(
                name="missing_timestamps",
                status=ValidationStatus.WARNING,
                message="dataset has missing expected daily timestamps",
                observed_value=missing_count,
            )
        return ValidationCheck(
            name="missing_timestamps",
            status=ValidationStatus.PASSED,
            message="dataset has expected daily timestamp coverage",
            observed_value=0,
        )

    def _check_nan_ratio(self, frame: pd.DataFrame, column: str) -> ValidationCheck:
        nan_ratio = float(frame[column].isna().mean())
        if nan_ratio > self.max_nan_ratio:
            return ValidationCheck(
                name=f"nan_ratio_{column}",
                status=ValidationStatus.FAILED,
                message=f"{column} NaN ratio exceeds threshold",
                observed_value=nan_ratio,
            )
        status = ValidationStatus.WARNING if nan_ratio > 0 else ValidationStatus.PASSED
        message = f"{column} contains NaNs" if nan_ratio > 0 else f"{column} contains no NaNs"
        return ValidationCheck(
            name=f"nan_ratio_{column}",
            status=status,
            message=message,
            observed_value=nan_ratio,
        )

    def _aggregate_status(self, checks: list[ValidationCheck]) -> ValidationStatus:
        if any(check.status == ValidationStatus.FAILED for check in checks):
            return ValidationStatus.FAILED
        if any(check.status == ValidationStatus.WARNING for check in checks):
            return ValidationStatus.WARNING
        return ValidationStatus.PASSED
