"""Convert vendor-shaped data into canonical internal schemas."""

from __future__ import annotations

import pandas as pd

from src.data.contracts import DataRequest, DataType


class OHLCVStandardizer:
    """Standardize source OHLCV data into the internal daily bar schema."""

    column_aliases = {
        "date": "timestamp",
        "datetime": "timestamp",
        "timestamp": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    canonical_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "source",
        "frequency",
    ]

    def transform(self, frame: pd.DataFrame, request: DataRequest) -> pd.DataFrame:
        """Return OHLCV data using canonical column names and deterministic ordering."""

        if request.data_type is not DataType.OHLCV:
            msg = "OHLCVStandardizer requires an ohlcv request"
            raise ValueError(msg)

        renamed = frame.rename(
            columns={col: self.column_aliases.get(col.lower(), col) for col in frame}
        )
        missing = {"timestamp", "open", "high", "low", "close", "volume"} - set(renamed.columns)
        if missing:
            msg = (
                "OHLCV data is missing source columns required for standardization: "
                f"{sorted(missing)}"
            )
            raise ValueError(msg)

        standardized = renamed.copy()
        standardized["timestamp"] = pd.to_datetime(
            standardized["timestamp"], utc=True
        ).dt.normalize()
        standardized["symbol"] = request.normalized_symbol
        standardized["source"] = request.source.value
        standardized["frequency"] = request.frequency.value
        return standardized[self.canonical_columns].sort_values("timestamp").reset_index(drop=True)


class MacroStandardizer:
    """Standardize macro observations into the internal macro schema."""

    canonical_columns = ["timestamp", "value", "series_id", "source", "frequency"]

    def transform(self, frame: pd.DataFrame, request: DataRequest) -> pd.DataFrame:
        """Return macro data using canonical column names and deterministic ordering."""

        if request.data_type is not DataType.MACRO:
            msg = "MacroStandardizer requires a macro request"
            raise ValueError(msg)

        working = frame.copy()
        if "DATE" in working.columns:
            working = working.rename(
                columns={"DATE": "timestamp", request.normalized_symbol: "value"}
            )
        elif "date" in working.columns:
            working = working.rename(columns={"date": "timestamp"})

        if "timestamp" not in working.columns:
            msg = "macro data is missing a timestamp/DATE column"
            raise ValueError(msg)

        if "value" not in working.columns:
            value_columns = [col for col in working.columns if col != "timestamp"]
            if len(value_columns) != 1:
                msg = "macro data must contain exactly one value column"
                raise ValueError(msg)
            working = working.rename(columns={value_columns[0]: "value"})

        standardized = working.copy()
        standardized["timestamp"] = pd.to_datetime(
            standardized["timestamp"], utc=True
        ).dt.normalize()
        standardized["value"] = pd.to_numeric(
            standardized["value"].replace(".", pd.NA), errors="coerce"
        )
        standardized["series_id"] = request.normalized_symbol
        standardized["source"] = request.source.value
        standardized["frequency"] = request.frequency.value
        return standardized[self.canonical_columns].sort_values("timestamp").reset_index(drop=True)
