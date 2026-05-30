from datetime import date

import pandas as pd
from src.data import DataAgent, DataFrequency, DataRequest, DataSource, DataType


class FakeYahooDownloader:
    def __init__(self) -> None:
        self.requests: list[DataRequest] = []

    def download(self, request: DataRequest) -> pd.DataFrame:
        self.requests.append(request)
        dates = pd.date_range(request.start_date, request.end_date, freq="B")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": range(1, len(dates) + 1),
                "High": range(2, len(dates) + 2),
                "Low": range(0, len(dates)),
                "Close": range(1, len(dates) + 1),
                "Volume": [100] * len(dates),
            }
        )


def _request(end: date) -> DataRequest:
    return DataRequest(
        symbol="SPY",
        data_type=DataType.OHLCV,
        source=DataSource.YFINANCE,
        start_date=date(2020, 1, 1),
        end_date=end,
        frequency=DataFrequency.DAILY,
    )


def test_data_agent_stores_processed_parquet_and_registry(tmp_path) -> None:
    downloader = FakeYahooDownloader()
    agent = DataAgent(
        project_root=tmp_path,
        downloaders={DataSource.YFINANCE.value: downloader},
    )

    frame = agent.load(_request(date(2020, 1, 3)))

    assert list(frame.columns) == [
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
    assert (tmp_path / "data" / "processed" / "ohlcv" / "SPY" / "1d.parquet").exists()
    assert (tmp_path / "data" / "external" / "registry" / "datasets.json").exists()
    assert downloader.requests[0].start_date == date(2020, 1, 1)


def test_data_agent_incremental_update_downloads_missing_period_only(tmp_path) -> None:
    downloader = FakeYahooDownloader()
    agent = DataAgent(
        project_root=tmp_path,
        downloaders={DataSource.YFINANCE.value: downloader},
    )

    agent.load(_request(date(2020, 1, 3)))
    frame = agent.load(_request(date(2020, 1, 7)))

    assert downloader.requests[1].start_date == date(2020, 1, 4)
    assert downloader.requests[1].end_date == date(2020, 1, 7)
    assert len(frame) == 5
