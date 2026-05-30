"""Tests for orchestration.utils.serialization."""

import json

import pandas as pd
from src.orchestration.utils.serialization import (
    dump_json,
    load_json,
    load_parquet,
    load_series_parquet,
)


def test_load_json_missing(tmp_path):
    assert load_json(tmp_path / "nonexistent.json") is None


def test_load_json_valid(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{"key": "value", "n": 42}')
    result = load_json(p)
    assert result == {"key": "value", "n": 42}


def test_load_json_list(tmp_path):
    p = tmp_path / "list.json"
    p.write_text('[1, 2, 3]')
    result = load_json(p)
    assert result == [1, 2, 3]


def test_load_json_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not-json{")
    assert load_json(p) is None


def test_load_parquet_missing(tmp_path):
    assert load_parquet(tmp_path / "nonexistent.parquet") is None


def test_load_parquet_valid(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    result = load_parquet(p)
    assert result is not None
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 3


def test_load_series_parquet(tmp_path):
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
    p = tmp_path / "series.parquet"
    df.to_parquet(p)
    s = load_series_parquet(p)
    assert s is not None
    assert len(s) == 3


def test_load_series_parquet_named_column(tmp_path):
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    p = tmp_path / "series.parquet"
    df.to_parquet(p)
    s = load_series_parquet(p, column="b")
    assert s is not None
    assert s.iloc[0] == 2.0


def test_dump_json(tmp_path):
    p = tmp_path / "sub" / "out.json"
    dump_json({"x": 1, "y": [1, 2]}, p)
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded["x"] == 1
    assert loaded["y"] == [1, 2]


def test_dump_json_creates_parents(tmp_path):
    p = tmp_path / "a" / "b" / "c" / "file.json"
    dump_json({}, p)
    assert p.exists()
