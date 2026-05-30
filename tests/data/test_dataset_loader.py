from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from src.data import (
    DatasetColumnError,
    DatasetNotFoundError,
    DatasetQuery,
    DatasetSchemaVersionError,
    MultipleDatasetsMatchedError,
    load_dataset,
    resolve_dataset_manifest,
)
from src.data.manifest import DatasetManifest
from src.data.registry import DatasetRegistry


def test_load_dataset_reads_registered_parquet(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "processed" / "ohlcv" / "SPY" / "1d.parquet"
    dataset_path.parent.mkdir(parents=True)
    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"], utc=True),
            "close": [100.0],
            "symbol": ["SPY"],
        }
    )
    expected.to_parquet(dataset_path, engine="pyarrow", index=False)
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    DatasetRegistry(registry_path).register(_manifest(symbol="SPY", storage_path=str(dataset_path)))

    loaded = load_dataset(
        symbol="SPY",
        data_type="ohlcv",
        frequency="1d",
        source="yfinance",
        registry_path=registry_path,
    )

    pd.testing.assert_frame_equal(loaded, expected)


def test_load_dataset_accepts_dataset_query(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "processed" / "ohlcv" / "SPY" / "1d.parquet"
    dataset_path.parent.mkdir(parents=True)
    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"], utc=True),
            "close": [100.0],
            "symbol": ["SPY"],
        }
    )
    expected.to_parquet(dataset_path, engine="pyarrow", index=False)
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    DatasetRegistry(registry_path).register(_manifest(symbol="SPY", storage_path=str(dataset_path)))

    loaded = load_dataset(
        DatasetQuery(
            dataset_family="ohlcv",
            symbol="spy",
            frequency="1d",
            source="yfinance",
            required_columns=("timestamp", "close"),
        ),
        registry_path=registry_path,
    )

    pd.testing.assert_frame_equal(loaded, expected)


def test_load_dataset_raises_clear_error_when_missing(tmp_path: Path) -> None:
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    DatasetRegistry(registry_path)

    with pytest.raises(DatasetNotFoundError, match="dataset not found in registry"):
        load_dataset(
            symbol="SPY",
            data_type="ohlcv",
            frequency="1d",
            source="yfinance",
            registry_path=registry_path,
        )


def test_resolve_dataset_manifest_rejects_ambiguous_matches(tmp_path: Path) -> None:
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    registry = DatasetRegistry(registry_path)
    registry.register(_manifest(symbol="SPY", request_hash="first-hash"))
    registry.register(_manifest(symbol="SPY", request_hash="second-hash"))

    with pytest.raises(MultipleDatasetsMatchedError, match="multiple datasets matched"):
        resolve_dataset_manifest(
            DatasetQuery(
                dataset_family="ohlcv",
                symbol="SPY",
                frequency="1d",
                source="yfinance",
            ),
            registry_path=registry_path,
        )


def test_resolve_dataset_manifest_rejects_schema_incompatibility(tmp_path: Path) -> None:
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    DatasetRegistry(registry_path).register(_manifest(symbol="SPY", schema_version="v1"))

    with pytest.raises(DatasetSchemaVersionError, match="incompatible schema version"):
        resolve_dataset_manifest(
            DatasetQuery(
                dataset_family="ohlcv",
                symbol="SPY",
                frequency="1d",
                source="yfinance",
                schema_version="v2",
            ),
            registry_path=registry_path,
        )


def test_load_dataset_rejects_missing_required_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data" / "processed" / "ohlcv" / "SPY" / "1d.parquet"
    dataset_path.parent.mkdir(parents=True)
    pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01"], utc=True)}).to_parquet(
        dataset_path,
        engine="pyarrow",
        index=False,
    )
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    DatasetRegistry(registry_path).register(_manifest(symbol="SPY", storage_path=str(dataset_path)))

    with pytest.raises(DatasetColumnError, match="missing required columns"):
        load_dataset(
            symbol="SPY",
            data_type="ohlcv",
            frequency="1d",
            source="yfinance",
            required_columns=("timestamp", "close"),
            registry_path=registry_path,
        )


def test_registry_query_helpers_filter_exact_matches(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path / "data" / "external" / "registry" / "datasets.json")
    registry.register(_manifest(symbol="SPY", request_hash="spy-hash"))
    registry.register(_manifest(symbol="QQQ", request_hash="qqq-hash"))

    assert [manifest.symbol for manifest in registry.find_by_symbol("spy")] == ["SPY"]
    assert [manifest.symbol for manifest in registry.find_by_source("yfinance")] == [
        "QQQ",
        "SPY",
    ]
    assert registry.find(symbol="SPY", source="fred") == []
    query_matches = registry.query_datasets(DatasetQuery(symbol="spy"))
    assert [manifest.symbol for manifest in query_matches] == ["SPY"]


def _manifest(
    *,
    symbol: str,
    storage_path: str = "data/processed/ohlcv/SPY/1d.parquet",
    request_hash: str = "request-hash",
    schema_version: str = "v1",
) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=f"ohlcv_yfinance_{symbol}_1d",
        dataset_name="daily_prices",
        symbol=symbol,
        data_type="ohlcv",
        source="yfinance",
        frequency="1d",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        schema_version=schema_version,
        storage_path=storage_path,
        row_count=1,
        created_at=datetime(2020, 1, 4, tzinfo=UTC),
        request_hash=request_hash,
    )
