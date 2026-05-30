"""Tests for src/experiments/orchestrator.py.

Data loading is patched so tests run offline and deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments.orchestrator import (
    ExperimentRun,
    _write_normalized_config,
    _write_raw_config,
    run_experiment_from_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(tickers: list[str] = None, n: int = 500) -> pd.DataFrame:
    tickers = tickers or ["SPY", "QQQ"]
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    data = rng.lognormal(0.0002, 0.01, size=(n, len(tickers))).cumprod(axis=0) * 100
    return pd.DataFrame(data, index=idx, columns=tickers)


def _make_raw_df(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return per-symbol DataFrame as load_dataset would produce."""
    return pd.DataFrame({"close": prices[ticker]})


def _make_load_universe_patch(prices: pd.DataFrame):
    """Return a universe dict mock suitable for align_prices()."""
    def _mock(symbols, **kwargs):
        return {sym: _make_raw_df(prices, sym) for sym in symbols if sym in prices.columns}
    return _mock


_MINIMAL_CFG = {
    "name": "orch_test",
    "universe": {"tickers": ["SPY", "QQQ"]},
    "date_range": {"start": "2015-01-01", "end": "2017-12-31"},
    "strategy": {"type": "EqualWeight", "parameters": {"rebalance_freq": "ME"}},
}


def _write_cfg(tmp_path: Path, cfg: dict, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _write_raw_config
# ---------------------------------------------------------------------------


def test_write_raw_config_copies_yaml(tmp_path: Path) -> None:
    source = tmp_path / "cfg.yaml"
    source.write_text("name: test\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_raw_config({}, out_dir, source)
    dest = out_dir / "raw_config.yaml"
    assert dest.exists()
    assert dest.read_text() == "name: test\n"


def test_write_raw_config_preserves_extension_json(tmp_path: Path) -> None:
    source = tmp_path / "cfg.json"
    source.write_text('{"name": "test"}', encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_raw_config({}, out_dir, source)
    assert (out_dir / "raw_config.json").exists()


# ---------------------------------------------------------------------------
# _write_normalized_config
# ---------------------------------------------------------------------------


def test_write_normalized_config_creates_json(tmp_path: Path) -> None:
    _write_normalized_config({"version": "1", "name": "x"}, tmp_path)
    dest = tmp_path / "normalized_config.json"
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["name"] == "x"


def test_write_normalized_config_sorted_keys(tmp_path: Path) -> None:
    cfg = {"z_key": 1, "a_key": 2}
    _write_normalized_config(cfg, tmp_path)
    text = (tmp_path / "normalized_config.json").read_text()
    assert text.index('"a_key"') < text.index('"z_key"')


# ---------------------------------------------------------------------------
# run_experiment_from_config
# ---------------------------------------------------------------------------


def _run_with_mock(config_path: Path, prices: pd.DataFrame) -> ExperimentRun:
    with patch("src.experiments.orchestrator.load_universe", side_effect=_make_load_universe_patch(prices)):
        return run_experiment_from_config(config_path)


def test_run_returns_experiment_run(tmp_path: Path) -> None:
    prices = _make_prices(["SPY", "QQQ"])
    p = _write_cfg(tmp_path, _MINIMAL_CFG)
    run = _run_with_mock(p, prices)
    assert isinstance(run, ExperimentRun)


def test_run_output_path_is_directory(tmp_path: Path) -> None:
    prices = _make_prices(["SPY", "QQQ"])
    cfg = {**_MINIMAL_CFG, "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(tmp_path / "registry.json")}}
    p = _write_cfg(tmp_path, cfg)
    run = _run_with_mock(p, prices)
    assert run.output_path.is_dir()


def test_run_saves_raw_config(tmp_path: Path) -> None:
    prices = _make_prices(["SPY", "QQQ"])
    cfg = {**_MINIMAL_CFG, "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(tmp_path / "registry.json")}}
    p = _write_cfg(tmp_path, cfg)
    run = _run_with_mock(p, prices)
    assert (run.output_path / "raw_config.yaml").exists()


def test_run_saves_normalized_config(tmp_path: Path) -> None:
    prices = _make_prices(["SPY", "QQQ"])
    cfg = {**_MINIMAL_CFG, "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(tmp_path / "registry.json")}}
    p = _write_cfg(tmp_path, cfg)
    run = _run_with_mock(p, prices)
    assert (run.output_path / "normalized_config.json").exists()


def test_run_registers_experiment(tmp_path: Path) -> None:
    import json as _json

    prices = _make_prices(["SPY", "QQQ"])
    registry_path = tmp_path / "registry.json"
    cfg = {**_MINIMAL_CFG, "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(registry_path)}}
    p = _write_cfg(tmp_path, cfg)
    _run_with_mock(p, prices)
    assert registry_path.exists()
    entries = _json.loads(registry_path.read_text())
    assert len(entries) >= 1


def test_run_experiment_result_has_metrics(tmp_path: Path) -> None:
    prices = _make_prices(["SPY", "QQQ"])
    cfg = {**_MINIMAL_CFG, "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(tmp_path / "r.json")}}
    p = _write_cfg(tmp_path, cfg)
    run = _run_with_mock(p, prices)
    assert "sharpe_ratio" in run.experiment_result.metrics


def test_run_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_experiment_from_config(tmp_path / "missing.yaml")


def test_run_invalid_config_raises(tmp_path: Path) -> None:
    bad_cfg = {"name": "bad"}  # missing universe, date_range, strategy
    p = _write_cfg(tmp_path, bad_cfg)
    with pytest.raises(ValueError):
        with patch("src.experiments.orchestrator.load_universe"):
            run_experiment_from_config(p)


def test_run_no_walk_forward_when_type_none(tmp_path: Path) -> None:
    prices = _make_prices(["SPY", "QQQ"])
    cfg = {**_MINIMAL_CFG, "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(tmp_path / "r.json")}}
    p = _write_cfg(tmp_path, cfg)
    run = _run_with_mock(p, prices)
    assert run.walk_forward is None


def test_run_with_validation_produces_walk_forward(tmp_path: Path) -> None:
    # 5 years of data — enough for train=24mo + test=12mo with room for splits
    prices = _make_prices(["SPY", "QQQ"], n=1260)
    cfg = {
        **_MINIMAL_CFG,
        "date_range": {"start": "2015-01-01", "end": "2019-12-31"},
        "validation": {
            "type": "rolling",
            "parameters": {"train_months": 24, "test_months": 12},
        },
        "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(tmp_path / "r.json")},
    }
    p = _write_cfg(tmp_path, cfg)
    run = _run_with_mock(p, prices)
    assert run.walk_forward is not None
    assert run.walk_forward.n_splits > 0


def test_run_json_config(tmp_path: Path) -> None:
    prices = _make_prices(["SPY", "QQQ"])
    p = tmp_path / "cfg.json"
    cfg_with_output = {**_MINIMAL_CFG, "output": {"base_dir": str(tmp_path / "results"), "registry_path": str(tmp_path / "r.json")}}
    p.write_text(json.dumps(cfg_with_output), encoding="utf-8")
    run = _run_with_mock(p, prices)
    assert isinstance(run, ExperimentRun)
    assert (run.output_path / "raw_config.json").exists()
