"""Downloader interfaces."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.data.contracts import DataRequest


class DataDownloader(Protocol):
    """External source adapter that returns source-shaped pandas data."""

    def download(self, request: DataRequest) -> pd.DataFrame:
        """Download data for the request."""
        ...
