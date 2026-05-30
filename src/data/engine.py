"""Profile-driven ingestion execution for Data Agent V1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from loguru import logger

from src.core import DateRange, Universe
from src.data.config import (
    DatasetProfileConfig,
    expand_profile_data_requests,
    load_data_agent_v1_config,
    load_dataset_profile_config,
)
from src.data.contracts import DataRequest, DataSource, DataType
from src.data.downloaders import DataDownloader, FredDownloader, YahooFinanceDownloader
from src.data.loaders.storage import DataStorage
from src.data.manifest import DatasetManifest, hash_request
from src.data.registry import DatasetRegistry
from src.data.transformers import MacroStandardizer, OHLCVStandardizer
from src.data.validators import DatasetValidator


class DatasetUpdateEngine:
    """Run dataset profiles through the existing V1 ingestion components."""

    def __init__(
        self,
        project_root: Path | str = ".",
        downloaders: dict[str, DataDownloader] | None = None,
        storage: DataStorage | None = None,
        registry: DatasetRegistry | None = None,
        validator: DatasetValidator | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.storage = storage or DataStorage(self.project_root)
        self.validator = validator or DatasetValidator()
        self.downloaders = downloaders or {
            DataSource.YFINANCE.value: YahooFinanceDownloader(),
            DataSource.FRED.value: FredDownloader(),
        }
        self.registry = registry
        self.ohlcv_standardizer = OHLCVStandardizer()
        self.macro_standardizer = MacroStandardizer()

    def run_profile(self, profile_path: str | Path) -> tuple[DatasetManifest, ...]:
        """Execute one dataset profile and register generated manifests."""

        logger.info("[LOAD PROFILE] {}", profile_path)
        resolved_profile_path = self._resolve_path(profile_path)
        profile = load_dataset_profile_config(resolved_profile_path)
        config_path = self._resolve_config_path(profile, resolved_profile_path)
        config = load_data_agent_v1_config(config_path)
        if profile.schema_version != config.schema_version:
            msg = (
                "dataset profile schema_version must match ingestion config "
                f"schema_version: {profile.schema_version} != {config.schema_version}"
            )
            raise ValueError(msg)
        registry = self.registry or DatasetRegistry(self.project_root / config.storage.registry)

        universe = self._load_profile_universe(profile, resolved_profile_path)
        date_range = self._load_profile_date_range(profile)

        logger.bind(
            dataset_name=profile.name,
            universe=universe.name,
            symbols=len(universe.symbols),
        ).info("[BUILD REQUESTS]")
        requests = expand_profile_data_requests(
            profile=profile,
            config=config,
            universe=universe,
            date_range=date_range,
        )

        manifests: list[DatasetManifest] = []
        for request in requests:
            request_hash = hash_request(request, schema_version=profile.schema_version)
            if registry.exists(request_hash):
                logger.bind(
                    dataset_id=request.dataset_id,
                    request_hash=request_hash,
                ).info("[SKIP] dataset already exists")
                manifests.extend(registry.find(request_hash=request_hash))
                continue

            manifest = self._run_request(
                request=request,
                profile=profile,
                request_hash=request_hash,
            )
            registry.register(manifest)
            logger.bind(
                dataset_id=request.dataset_id,
                request_hash=request_hash,
            ).info("[REGISTER]")
            manifests.append(manifest)

        return tuple(manifests)

    def _run_request(
        self,
        *,
        request: DataRequest,
        profile: DatasetProfileConfig,
        request_hash: str,
    ) -> DatasetManifest:
        downloader = self.downloaders.get(request.source.value)
        if downloader is None:
            msg = f"no downloader registered for source {request.source.value}"
            raise ValueError(msg)

        logger.bind(dataset_id=request.dataset_id).info("[DOWNLOAD]")
        raw = downloader.download(request)

        standardized = self._standardize(raw, request)

        logger.bind(dataset_id=request.dataset_id).info("[VALIDATE]")
        report = self.validator.validate(standardized, request)
        if report.is_failed:
            msg = f"validation failed for {request.dataset_id}"
            raise ValueError(msg)

        logger.bind(dataset_id=request.dataset_id).info("[SAVE]")
        self.storage.write_raw(raw, request)
        storage_path = self.storage.write_processed(standardized, request)
        self.storage.validation_report_path(request).write_text(
            report.model_dump_json(indent=2) + "\n"
        )

        return DatasetManifest(
            dataset_id=request.dataset_id,
            dataset_name=profile.name,
            symbol=request.normalized_symbol,
            data_type=request.data_type.value,
            source=request.source.value,
            frequency=request.frequency.value,
            start_date=request.start_date,
            end_date=request.end_date,
            schema_version=profile.schema_version,
            storage_path=str(storage_path),
            row_count=len(standardized),
            request_hash=request_hash,
        )

    def _standardize(self, frame: pd.DataFrame, request: DataRequest) -> pd.DataFrame:
        if request.data_type is DataType.OHLCV:
            return self.ohlcv_standardizer.transform(frame, request)
        return self.macro_standardizer.transform(frame, request)

    def _load_profile_universe(
        self,
        profile: DatasetProfileConfig,
        profile_path: Path,
    ) -> Universe:
        if profile.universe is None:
            msg = "dataset profile must define a universe path"
            raise ValueError(msg)
        universe_path = self._resolve_relative_path(profile.universe, profile_path.parent)
        raw = self._load_yaml(universe_path)
        universe_raw = raw.get("universe", raw)
        symbols = universe_raw.get("symbols", [])
        if not symbols:
            msg = f"universe config contains no symbols: {universe_path}"
            raise ValueError(msg)
        return Universe(
            name=str(universe_raw["name"]),
            symbols=tuple(str(symbol) for symbol in symbols),
            description=universe_raw.get("description"),
        )

    def _load_profile_date_range(self, profile: DatasetProfileConfig) -> DateRange:
        if profile.date_range is None:
            msg = "dataset profile must define a date_range"
            raise ValueError(msg)
        return DateRange(start=profile.date_range.start, end=profile.date_range.end)

    def _resolve_config_path(
        self,
        profile: DatasetProfileConfig,
        profile_path: Path,
    ) -> Path:
        return self._resolve_relative_path(profile.ingestion_config, profile_path.parent)

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def _resolve_relative_path(self, path: Path, base_dir: Path) -> Path:
        if path.is_absolute():
            return path
        project_relative = (self.project_root / path).resolve()
        if project_relative.exists():
            return project_relative
        return (base_dir / path).resolve()

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML config must be a mapping: {path}")
        return loaded
