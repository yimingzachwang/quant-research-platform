from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from src.data import DataRequest, DatasetUpdateEngine, hash_request
from src.data.registry import DatasetRegistry


class FakeYahooDownloader:
    def __init__(self, duplicate_timestamps: bool = False) -> None:
        self.duplicate_timestamps = duplicate_timestamps
        self.requests: list[DataRequest] = []

    def download(self, request: DataRequest) -> pd.DataFrame:
        self.requests.append(request)
        if self.duplicate_timestamps:
            dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
        else:
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


def test_engine_run_profile_persists_dataset_and_manifest(tmp_path: Path) -> None:
    profile_path = _write_profile_project(tmp_path)
    downloader = FakeYahooDownloader()

    manifests = DatasetUpdateEngine(
        project_root=tmp_path,
        downloaders={"yfinance": downloader},
    ).run_profile(profile_path)

    assert len(manifests) == 2
    assert [manifest.symbol for manifest in manifests] == ["SPY", "QQQ"]
    assert all(manifest.dataset_name == "daily_prices" for manifest in manifests)
    assert all(manifest.schema_version == "v1" for manifest in manifests)
    assert all(manifest.row_count == 3 for manifest in manifests)
    assert (tmp_path / "data" / "processed" / "ohlcv" / "SPY" / "1d.parquet").exists()
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    assert registry_path.exists()
    assert len(DatasetRegistry(registry_path).load_all()) == 2


def test_engine_skips_duplicate_registered_request(tmp_path: Path) -> None:
    profile_path = _write_profile_project(tmp_path)
    downloader = FakeYahooDownloader()
    engine = DatasetUpdateEngine(
        project_root=tmp_path,
        downloaders={"yfinance": downloader},
    )

    first_run = engine.run_profile(profile_path)
    second_run = engine.run_profile(profile_path)

    assert len(first_run) == 2
    assert len(second_run) == 2
    assert len(downloader.requests) == 2
    assert [manifest.request_hash for manifest in second_run] == [
        manifest.request_hash for manifest in first_run
    ]


def test_registry_creates_missing_file_and_persists_manifest(tmp_path: Path) -> None:
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    registry = DatasetRegistry(registry_path)

    assert registry_path.exists()
    assert registry.load_all() == []


def test_engine_failed_validation_does_not_register_manifest(tmp_path: Path) -> None:
    profile_path = _write_profile_project(tmp_path)
    downloader = FakeYahooDownloader(duplicate_timestamps=True)
    engine = DatasetUpdateEngine(
        project_root=tmp_path,
        downloaders={"yfinance": downloader},
    )

    with pytest.raises(ValueError, match="validation failed"):
        engine.run_profile(profile_path)

    registry = DatasetRegistry(tmp_path / "data" / "external" / "registry" / "datasets.json")
    assert registry.load_all() == []
    assert not (tmp_path / "data" / "processed" / "ohlcv" / "SPY" / "1d.parquet").exists()


def test_hash_request_includes_schema_version(tmp_path: Path) -> None:
    profile_path = _write_profile_project(tmp_path)
    manifest = DatasetUpdateEngine(
        project_root=tmp_path,
        downloaders={"yfinance": FakeYahooDownloader()},
    ).run_profile(profile_path)[0]
    request = DataRequest(
        symbol=manifest.symbol,
        data_type=manifest.data_type,
        source=manifest.source,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        frequency=manifest.frequency,
    )

    assert manifest.request_hash == hash_request(request, schema_version="v1")
    assert manifest.request_hash != hash_request(request, schema_version="v2")


def _write_profile_project(project_root: Path) -> Path:
    config_dir = project_root / "configs" / "data"
    universe_dir = project_root / "configs" / "universes"
    config_dir.mkdir(parents=True)
    universe_dir.mkdir(parents=True)

    (config_dir / "data_agent_v1.yaml").write_text("""
name: data_agent_v1
schema_version: v1
supported_data_types:
  - ohlcv
supported_sources:
  ohlcv:
    - yfinance
frequency: 1d
storage:
  raw: data/raw
  processed: data/processed
  features: data/features
  metadata: data/external/metadata
  registry: data/external/registry/datasets.json
validation:
  max_nan_ratio: 0.05
canonical_schemas:
  ohlcv:
    - timestamp
    - open
    - high
    - low
    - close
    - adjusted_close
    - volume
    - symbol
    - source
    - frequency
""".lstrip())
    (universe_dir / "test_etfs.yaml").write_text("""
name: test_etfs
symbols:
  - SPY
  - QQQ
""".lstrip())
    profile_path = config_dir / "daily_prices.yaml"
    profile_path.write_text("""
name: daily_prices
schema_version: v1
ingestion_config: configs/data/data_agent_v1.yaml
universe: configs/universes/test_etfs.yaml
date_range:
  start: "2020-01-01"
  end: "2020-01-03"
data_type: ohlcv
source: yfinance
frequency: 1d
""".lstrip())
    return profile_path
