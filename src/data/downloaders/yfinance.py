"""Yahoo Finance downloader for daily ETF OHLCV bars."""

from __future__ import annotations

from datetime import timedelta
from time import sleep

import pandas as pd
from loguru import logger

from src.data.contracts import DataRequest, DataSource, DataType


class YahooFinanceDownloader:
    """Download daily OHLCV ETF data from yfinance."""

    def __init__(self, retries: int = 3, backoff_seconds: float = 1.0) -> None:
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def download(self, request: DataRequest) -> pd.DataFrame:
        """Download source-shaped Yahoo Finance bars."""

        if request.data_type is not DataType.OHLCV or request.source is not DataSource.YFINANCE:
            msg = "YahooFinanceDownloader only supports ohlcv/yfinance requests"
            raise ValueError(msg)

        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment guard
            msg = "yfinance is required for YahooFinanceDownloader"
            raise RuntimeError(msg) from exc

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.bind(
                    source=request.source.value,
                    symbol=request.normalized_symbol,
                    attempt=attempt,
                ).info("downloading Yahoo Finance OHLCV")
                frame = yf.download(
                    tickers=request.normalized_symbol,
                    start=request.start_date.isoformat(),
                    end=(request.end_date + timedelta(days=1)).isoformat(),
                    interval=request.frequency.value,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(0)
                frame = frame.reset_index()
                frame["Date"] = pd.to_datetime(frame["Date"], utc=True)
                return frame
            except Exception as exc:  # pragma: no cover - exercised only on network failures
                last_error = exc
                logger.bind(attempt=attempt, error=str(exc)).warning("Yahoo download failed")
                if attempt < self.retries:
                    sleep(self.backoff_seconds * attempt)

        msg = f"failed to download Yahoo Finance data for {request.normalized_symbol}"
        raise RuntimeError(msg) from last_error
