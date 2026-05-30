from datetime import UTC, date, datetime
from pathlib import Path

from scripts.list_datasets import format_dataset_table, load_filtered_manifests
from src.data.manifest import DatasetManifest
from src.data.registry import DatasetRegistry


def test_format_dataset_table_handles_empty_registry() -> None:
    assert format_dataset_table([]) == "No datasets registered."


def test_list_datasets_formats_clean_table_and_filters(tmp_path: Path) -> None:
    registry_path = tmp_path / "data" / "external" / "registry" / "datasets.json"
    registry = DatasetRegistry(registry_path)
    registry.register(_manifest(symbol="SPY", row_count=4024, request_hash="spy-hash"))
    registry.register(_manifest(symbol="QQQ", row_count=4024, request_hash="qqq-hash"))

    all_output = format_dataset_table(load_filtered_manifests(registry_path=registry_path))
    filtered_output = format_dataset_table(
        load_filtered_manifests(registry_path=registry_path, symbol="SPY")
    )

    assert "DATASET ID" in all_output
    assert "ohlcv_yfinance_SPY_1d" in all_output
    assert "ohlcv_yfinance_QQQ_1d" in all_output
    assert "4024" in all_output
    assert "ohlcv_yfinance_SPY_1d" in filtered_output
    assert "ohlcv_yfinance_QQQ_1d" not in filtered_output


def _manifest(*, symbol: str, row_count: int, request_hash: str) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=f"ohlcv_yfinance_{symbol}_1d",
        dataset_name="daily_prices",
        symbol=symbol,
        data_type="ohlcv",
        source="yfinance",
        frequency="1d",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        schema_version="v1",
        storage_path=f"data/processed/ohlcv/{symbol}/1d.parquet",
        row_count=row_count,
        created_at=datetime(2020, 1, 4, tzinfo=UTC),
        request_hash=request_hash,
    )
