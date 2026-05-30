"""Unified internal loading API for research workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.contracts import DataRequest, DataSource
from src.data.downloaders import DataDownloader, FredDownloader, YahooFinanceDownloader
from src.data.loaders.storage import DataStorage
from src.data.registry import JsonDatasetRegistry
from src.data.update_engine import DatasetUpdateEngine
from src.data.validators import DatasetValidator


class DataAgent:
    """Deterministic facade for loading and updating internal datasets."""

    def __init__(
        self,
        project_root: Path | str = ".",
        downloaders: dict[str, DataDownloader] | None = None,
        validator: DatasetValidator | None = None,
    ) -> None:
        self.storage = DataStorage(project_root)
        self.registry = JsonDatasetRegistry(project_root)
        self.validator = validator or DatasetValidator()
        self.downloaders = downloaders or {
            DataSource.YFINANCE.value: YahooFinanceDownloader(),
            DataSource.FRED.value: FredDownloader(),
        }
        self.update_engine = DatasetUpdateEngine(
            storage=self.storage,
            registry=self.registry,
            validator=self.validator,
            downloaders=self.downloaders,
        )

    def load(self, request: DataRequest, refresh: bool = False) -> pd.DataFrame:
        """Load data through the internal API, optionally refreshing from source."""

        logger.bind(dataset_id=request.dataset_id, refresh=refresh).info("loading via data agent")
        if refresh or not self.storage.has_processed(request):
            return self.update(request)

        frame = self.storage.load_processed(request)
        timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.date
        covers_request = (
            timestamps.min() <= request.start_date and timestamps.max() >= request.end_date
        )
        if not covers_request:
            return self.update(request)
        return frame.loc[
            (timestamps >= request.start_date) & (timestamps <= request.end_date)
        ].reset_index(drop=True)

    def update(self, request: DataRequest) -> pd.DataFrame:
        """Incrementally update a dataset and return the requested date slice."""

        return self.update_engine.update(request)
