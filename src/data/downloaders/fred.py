"""FRED downloader for daily macroeconomic series."""

from __future__ import annotations

from time import sleep

import pandas as pd
from loguru import logger

from src.data.contracts import DataRequest, DataSource, DataType


class FredDownloader:
    """Download macro series from FRED using the public CSV endpoint."""

    def __init__(self, retries: int = 3, backoff_seconds: float = 1.0) -> None:
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def download(self, request: DataRequest) -> pd.DataFrame:
        """Download source-shaped FRED observations."""

        if request.data_type is not DataType.MACRO or request.source is not DataSource.FRED:
            msg = "FredDownloader only supports macro/fred requests"
            raise ValueError(msg)

        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={request.normalized_symbol}"
            f"&cosd={request.start_date.isoformat()}"
            f"&coed={request.end_date.isoformat()}"
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.bind(
                    source=request.source.value,
                    symbol=request.normalized_symbol,
                    attempt=attempt,
                ).info("downloading FRED series")
                frame = pd.read_csv(url)
                frame["DATE"] = pd.to_datetime(frame["DATE"], utc=True)
                return frame
            except Exception as exc:  # pragma: no cover - exercised only on network failures
                last_error = exc
                logger.bind(attempt=attempt, error=str(exc)).warning("FRED download failed")
                if attempt < self.retries:
                    sleep(self.backoff_seconds * attempt)

        msg = f"failed to download FRED series {request.normalized_symbol}"
        raise RuntimeError(msg) from last_error
