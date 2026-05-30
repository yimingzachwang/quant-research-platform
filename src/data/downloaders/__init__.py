"""Source-specific data downloaders."""

from src.data.downloaders.base import DataDownloader
from src.data.downloaders.fred import FredDownloader
from src.data.downloaders.yfinance import YahooFinanceDownloader

__all__ = ["DataDownloader", "FredDownloader", "YahooFinanceDownloader"]
