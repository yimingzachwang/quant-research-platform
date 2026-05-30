"""Tests for src/experiments/config_io.py."""

import json
from pathlib import Path

import pytest
import yaml
from src.experiments.config_io import load_config, normalize_config, validate_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: dict, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(content), encoding="utf-8")
    return p


def _write_json(tmp_path: Path, content: dict, name: str = "cfg.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(content), encoding="utf-8")
    return p


_MINIMAL = {
    "name": "test_exp",
    "universe": {"tickers": ["SPY", "QQQ"]},
    "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
    "strategy": {"type": "MomentumRotation"},
}


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_yaml(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, _MINIMAL)
    cfg = load_config(p)
    assert cfg["name"] == "test_exp"


def test_load_config_yml_extension(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, _MINIMAL, name="cfg.yml")
    cfg = load_config(p)
    assert isinstance(cfg, dict)


def test_load_config_json(tmp_path: Path) -> None:
    p = _write_json(tmp_path, _MINIMAL)
    cfg = load_config(p)
    assert cfg["name"] == "test_exp"


def test_load_config_accepts_string_path(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, _MINIMAL)
    cfg = load_config(str(p))
    assert isinstance(cfg, dict)


def test_load_config_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text("name = 'test'", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_config(p)


def test_load_config_non_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


def test_validate_minimal_config_passes() -> None:
    validate_config(_MINIMAL)


def test_validate_unknown_version_raises() -> None:
    cfg = {**_MINIMAL, "version": "99"}
    with pytest.raises(ValueError, match="version"):
        validate_config(cfg)


def test_validate_known_version_passes() -> None:
    cfg = {**_MINIMAL, "version": "1"}
    validate_config(cfg)


def test_validate_missing_name_raises() -> None:
    cfg = {k: v for k, v in _MINIMAL.items() if k != "name"}
    with pytest.raises(ValueError, match="name"):
        validate_config(cfg)


def test_validate_empty_name_raises() -> None:
    cfg = {**_MINIMAL, "name": "  "}
    with pytest.raises(ValueError, match="name"):
        validate_config(cfg)


def test_validate_missing_universe_raises() -> None:
    cfg = {k: v for k, v in _MINIMAL.items() if k != "universe"}
    with pytest.raises(ValueError, match="universe"):
        validate_config(cfg)


def test_validate_empty_tickers_raises() -> None:
    cfg = {**_MINIMAL, "universe": {"tickers": []}}
    with pytest.raises(ValueError, match="tickers"):
        validate_config(cfg)


def test_validate_date_range_start_ge_end_raises() -> None:
    cfg = {
        **_MINIMAL,
        "date_range": {"start": "2023-01-01", "end": "2020-01-01"},
    }
    with pytest.raises(ValueError, match="before"):
        validate_config(cfg)


def test_validate_unknown_strategy_type_raises() -> None:
    cfg = {**_MINIMAL, "strategy": {"type": "NeuralNet"}}
    with pytest.raises(ValueError, match="Unknown strategy"):
        validate_config(cfg)


def test_validate_invalid_validation_type_raises() -> None:
    cfg = {
        **_MINIMAL,
        "validation": {"type": "walk_forward"},
    }
    with pytest.raises(ValueError, match="Unknown validation"):
        validate_config(cfg)


def test_validate_negative_cost_raises() -> None:
    cfg = {**_MINIMAL, "execution": {"transaction_cost_bps": -1}}
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(cfg)


def test_validate_non_positive_train_months_raises() -> None:
    cfg = {
        **_MINIMAL,
        "validation": {
            "type": "rolling",
            "parameters": {"train_months": 0, "test_months": 12},
        },
    }
    with pytest.raises(ValueError, match="positive integer"):
        validate_config(cfg)


def test_validate_output_non_dict_raises() -> None:
    cfg = {**_MINIMAL, "output": "results/"}
    with pytest.raises(ValueError, match="mapping"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# normalize_config
# ---------------------------------------------------------------------------


def test_normalize_does_not_mutate_input() -> None:
    import copy
    original = copy.deepcopy(_MINIMAL)
    normalize_config(_MINIMAL)
    assert _MINIMAL == original


def test_normalize_sets_version() -> None:
    out = normalize_config(_MINIMAL)
    assert out["version"] == "1"


def test_normalize_sets_description() -> None:
    out = normalize_config(_MINIMAL)
    assert out["description"] == ""


def test_normalize_sets_tags_list() -> None:
    out = normalize_config(_MINIMAL)
    assert isinstance(out["tags"], list)


def test_normalize_sets_strategy_parameters() -> None:
    out = normalize_config(_MINIMAL)
    assert out["strategy"]["parameters"] == {}


def test_normalize_sets_validation_none() -> None:
    out = normalize_config(_MINIMAL)
    assert out["validation"]["type"] == "none"
    assert out["validation"]["parameters"] == {}


def test_normalize_sets_execution_cost_float() -> None:
    out = normalize_config(_MINIMAL)
    assert out["execution"]["transaction_cost_bps"] == 0.0
    assert isinstance(out["execution"]["transaction_cost_bps"], float)


def test_normalize_execution_cost_cast_to_float() -> None:
    cfg = {**_MINIMAL, "execution": {"transaction_cost_bps": 5}}
    out = normalize_config(cfg)
    assert isinstance(out["execution"]["transaction_cost_bps"], float)


def test_normalize_sets_output_defaults() -> None:
    out = normalize_config(_MINIMAL)
    assert "base_dir" in out["output"]
    assert "registry_path" in out["output"]
    assert out["output"]["register"] is True
    assert out["output"]["save_plots"] is True


def test_normalize_gap_days_for_rolling() -> None:
    cfg = {
        **_MINIMAL,
        "validation": {
            "type": "rolling",
            "parameters": {"train_months": 36, "test_months": 12},
        },
    }
    out = normalize_config(cfg)
    assert out["validation"]["parameters"]["gap_days"] == 0


def test_normalize_step_months_defaults_to_test_months() -> None:
    cfg = {
        **_MINIMAL,
        "validation": {
            "type": "rolling",
            "parameters": {"train_months": 36, "test_months": 6},
        },
    }
    out = normalize_config(cfg)
    assert out["validation"]["parameters"]["step_months"] == 6


def test_normalize_preserves_user_step_months() -> None:
    cfg = {
        **_MINIMAL,
        "validation": {
            "type": "rolling",
            "parameters": {"train_months": 36, "test_months": 12, "step_months": 3},
        },
    }
    out = normalize_config(cfg)
    assert out["validation"]["parameters"]["step_months"] == 3


def test_normalize_idempotent() -> None:
    out1 = normalize_config(_MINIMAL)
    out2 = normalize_config(out1)
    assert out1 == out2
