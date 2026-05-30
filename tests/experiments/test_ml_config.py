"""Tests for src/experiments/ml_config.py — F3 ML configuration layer."""

from __future__ import annotations

import pytest
from src.experiments.ml_config import (
    _VALID_FEATURE_TYPES,
    _VALID_LABEL_TYPES,
    _VALID_MODEL_TYPES,
    FeatureEntry,
    FeatureSpec,
    LabelSpec,
    MLExperimentSpec,
    ModelSpec,
    SignalSpec,
    build_ml_experiment_spec,
    ml_experiment_hash,
    normalize_ml_config,
    validate_ml_config,
)

# ---------------------------------------------------------------------------
# Minimal valid config fixture
# ---------------------------------------------------------------------------


def _minimal_v2_cfg() -> dict:
    return {
        "version": "2",
        "name": "test_ml",
        "universe": {"tickers": ["SPY"]},
        "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
        "features": {
            "ticker": "SPY",
            "entries": [{"name": "mom_20", "type": "momentum", "params": {"lookback": 20}}],
        },
        "labels": {"type": "forward_returns", "params": {"horizon": 5}},
        "model": {"type": "RidgeRegression", "params": {"alpha": 1.0}},
        "signal": {"type": "sign", "params": {}},
    }


# ---------------------------------------------------------------------------
# FeatureEntry
# ---------------------------------------------------------------------------


class TestFeatureEntry:
    def test_to_dict_round_trip(self):
        e = FeatureEntry(name="mom", type="momentum", params={"lookback": 20})
        d = e.to_dict()
        assert d == {"name": "mom", "type": "momentum", "params": {"lookback": 20}}

    def test_default_empty_params(self):
        e = FeatureEntry(name="ret", type="compute_returns")
        assert e.params == {}

    def test_params_are_copied_in_to_dict(self):
        params = {"lookback": 20}
        e = FeatureEntry(name="m", type="momentum", params=params)
        d = e.to_dict()
        d["params"]["lookback"] = 99
        assert e.params["lookback"] == 20


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------


class TestFeatureSpec:
    def test_to_dict(self):
        e = FeatureEntry(name="mom", type="momentum", params={"lookback": 20})
        fs = FeatureSpec(ticker="SPY", entries=[e])
        d = fs.to_dict()
        assert d["ticker"] == "SPY"
        assert len(d["entries"]) == 1

    def test_empty_entries_default(self):
        fs = FeatureSpec(ticker="SPY")
        assert fs.entries == []


# ---------------------------------------------------------------------------
# validate_ml_config — base section errors
# ---------------------------------------------------------------------------


class TestValidateMLConfigBase:
    def test_valid_minimal_config(self):
        validate_ml_config(_minimal_v2_cfg())  # no raise

    def test_missing_name(self):
        cfg = _minimal_v2_cfg()
        del cfg["name"]
        with pytest.raises(ValueError, match="non-empty string 'name'"):
            validate_ml_config(cfg)

    def test_empty_name(self):
        cfg = _minimal_v2_cfg()
        cfg["name"] = "   "
        with pytest.raises(ValueError, match="non-empty string 'name'"):
            validate_ml_config(cfg)

    def test_missing_universe(self):
        cfg = _minimal_v2_cfg()
        del cfg["universe"]
        with pytest.raises(ValueError, match="'universe' must be a mapping"):
            validate_ml_config(cfg)

    def test_empty_tickers(self):
        cfg = _minimal_v2_cfg()
        cfg["universe"]["tickers"] = []
        with pytest.raises(ValueError, match="'universe.tickers'"):
            validate_ml_config(cfg)

    def test_missing_date_range(self):
        cfg = _minimal_v2_cfg()
        del cfg["date_range"]
        with pytest.raises(ValueError, match="'date_range' must be a mapping"):
            validate_ml_config(cfg)

    def test_start_after_end(self):
        cfg = _minimal_v2_cfg()
        cfg["date_range"]["start"] = "2025-01-01"
        cfg["date_range"]["end"] = "2020-01-01"
        with pytest.raises(ValueError, match="must be before"):
            validate_ml_config(cfg)

    def test_invalid_validation_type(self):
        cfg = _minimal_v2_cfg()
        cfg["validation"] = {"type": "unknown"}
        with pytest.raises(ValueError, match="Unknown validation type"):
            validate_ml_config(cfg)

    def test_negative_transaction_cost(self):
        cfg = _minimal_v2_cfg()
        cfg["execution"] = {"transaction_cost_bps": -1}
        with pytest.raises(ValueError, match="non-negative"):
            validate_ml_config(cfg)


# ---------------------------------------------------------------------------
# validate_ml_config — features section errors
# ---------------------------------------------------------------------------


class TestValidateMLConfigFeatures:
    def test_missing_features(self):
        cfg = _minimal_v2_cfg()
        del cfg["features"]
        with pytest.raises(ValueError, match="'features' must be a mapping"):
            validate_ml_config(cfg)

    def test_missing_ticker(self):
        cfg = _minimal_v2_cfg()
        del cfg["features"]["ticker"]
        with pytest.raises(ValueError, match="'features.ticker'"):
            validate_ml_config(cfg)

    def test_empty_entries(self):
        cfg = _minimal_v2_cfg()
        cfg["features"]["entries"] = []
        with pytest.raises(ValueError, match="non-empty list"):
            validate_ml_config(cfg)

    def test_unknown_feature_type(self):
        cfg = _minimal_v2_cfg()
        cfg["features"]["entries"][0]["type"] = "unknown_feature"
        with pytest.raises(ValueError, match="unknown"):
            validate_ml_config(cfg)

    def test_missing_required_param(self):
        cfg = _minimal_v2_cfg()
        cfg["features"]["entries"][0]["params"] = {}
        with pytest.raises(ValueError, match="missing required key 'lookback'"):
            validate_ml_config(cfg)

    def test_entry_missing_name(self):
        cfg = _minimal_v2_cfg()
        del cfg["features"]["entries"][0]["name"]
        with pytest.raises(ValueError, match="non-empty string"):
            validate_ml_config(cfg)

    def test_all_feature_types_accepted(self):
        for ftype in sorted(_VALID_FEATURE_TYPES):
            cfg = _minimal_v2_cfg()
            params = {}
            if ftype == "momentum":
                params = {"lookback": 20}
            elif ftype in (
                "rolling_volatility", "rolling_zscore", "sma", "trend_strength",
                "downside_volatility", "bollinger_distance",
                "rolling_skewness", "rolling_autocorrelation",
                # Phase H-1 single-window types
                "trend_persistence", "breakout_strength", "drawdown_distance",
                "rolling_beta",
            ):
                params = {"window": 20}
            elif ftype == "ema":
                params = {"span": 20}
            elif ftype == "vol_compression":
                params = {"short_window": 21, "long_window": 63}
            elif ftype == "risk_adjusted_momentum":
                params = {"mom_window": 252}
            # vol_of_vol, vol_percentile, compute_returns have no required params
            cfg["features"]["entries"][0]["type"] = ftype
            cfg["features"]["entries"][0]["params"] = params
            validate_ml_config(cfg)  # no raise


# ---------------------------------------------------------------------------
# validate_ml_config — labels, model, signal errors
# ---------------------------------------------------------------------------


class TestValidateMLConfigLabels:
    def test_missing_labels(self):
        cfg = _minimal_v2_cfg()
        del cfg["labels"]
        with pytest.raises(ValueError, match="'labels' must be a mapping"):
            validate_ml_config(cfg)

    def test_unknown_label_type(self):
        cfg = _minimal_v2_cfg()
        cfg["labels"]["type"] = "mystery_label"
        with pytest.raises(ValueError, match="unknown"):
            validate_ml_config(cfg)

    def test_missing_horizon(self):
        cfg = _minimal_v2_cfg()
        cfg["labels"]["params"] = {}
        with pytest.raises(ValueError, match="missing required key 'horizon'"):
            validate_ml_config(cfg)

    def test_non_positive_horizon(self):
        cfg = _minimal_v2_cfg()
        cfg["labels"]["params"]["horizon"] = 0
        with pytest.raises(ValueError, match="positive integer"):
            validate_ml_config(cfg)

    def test_all_label_types_accepted(self):
        for ltype in sorted(_VALID_LABEL_TYPES):
            cfg = _minimal_v2_cfg()
            cfg["labels"]["type"] = ltype
            validate_ml_config(cfg)  # no raise (all have horizon param)


class TestValidateMLConfigModel:
    def test_missing_model(self):
        cfg = _minimal_v2_cfg()
        del cfg["model"]
        with pytest.raises(ValueError, match="'model' must be a mapping"):
            validate_ml_config(cfg)

    def test_unknown_model_type(self):
        cfg = _minimal_v2_cfg()
        cfg["model"]["type"] = "NeuralNet"
        with pytest.raises(ValueError, match="unknown"):
            validate_ml_config(cfg)

    def test_all_model_types_accepted(self):
        for mtype in sorted(_VALID_MODEL_TYPES):
            cfg = _minimal_v2_cfg()
            cfg["model"]["type"] = mtype
            cfg["model"]["params"] = {}
            validate_ml_config(cfg)  # no raise


class TestValidateMLConfigSignal:
    def test_missing_signal(self):
        cfg = _minimal_v2_cfg()
        del cfg["signal"]
        with pytest.raises(ValueError, match="'signal' must be a mapping"):
            validate_ml_config(cfg)

    def test_unknown_signal_type(self):
        cfg = _minimal_v2_cfg()
        cfg["signal"]["type"] = "magic_signal"
        with pytest.raises(ValueError, match="unknown"):
            validate_ml_config(cfg)

    def test_top_n_missing_n(self):
        cfg = _minimal_v2_cfg()
        cfg["signal"] = {"type": "top_n", "params": {}}
        with pytest.raises(ValueError, match="missing required key 'n'"):
            validate_ml_config(cfg)

    def test_long_short_missing_params(self):
        cfg = _minimal_v2_cfg()
        cfg["signal"] = {"type": "long_short", "params": {"n_long": 3}}
        with pytest.raises(ValueError, match="missing required key 'n_short'"):
            validate_ml_config(cfg)

    def test_all_signal_types_schema_valid(self):
        base_params = {
            "sign": {},
            "threshold": {},
            "top_n": {"n": 3},
            "long_short": {"n_long": 3, "n_short": 3},
            "normalize": {},
        }
        for stype, params in base_params.items():
            cfg = _minimal_v2_cfg()
            cfg["signal"] = {"type": stype, "params": params}
            validate_ml_config(cfg)  # no raise


# ---------------------------------------------------------------------------
# normalize_ml_config
# ---------------------------------------------------------------------------


class TestNormalizeMLConfig:
    def test_sets_version_2(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        assert out["version"] == "2"

    def test_fills_description_default(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        assert out["description"] == ""

    def test_fills_tags_default(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        assert out["tags"] == []

    def test_fills_validation_default(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        assert out["validation"]["type"] == "none"
        assert out["validation"]["parameters"] == {}

    def test_fills_execution_default(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        assert out["execution"]["transaction_cost_bps"] == 0.0

    def test_fills_output_defaults(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        assert "base_dir" in out["output"]
        assert "registry_path" in out["output"]

    def test_fills_feature_entry_params(self):
        cfg = _minimal_v2_cfg()
        del cfg["features"]["entries"][0]["params"]
        out = normalize_ml_config(cfg)
        assert isinstance(out["features"]["entries"][0]["params"], dict)

    def test_fills_labels_params(self):
        cfg = _minimal_v2_cfg()
        del cfg["labels"]["params"]
        out = normalize_ml_config(cfg)
        assert isinstance(out["labels"]["params"], dict)

    def test_fills_model_params(self):
        cfg = _minimal_v2_cfg()
        del cfg["model"]["params"]
        out = normalize_ml_config(cfg)
        assert isinstance(out["model"]["params"], dict)

    def test_fills_signal_params(self):
        cfg = _minimal_v2_cfg()
        del cfg["signal"]["params"]
        out = normalize_ml_config(cfg)
        assert isinstance(out["signal"]["params"], dict)

    def test_does_not_mutate_input(self):
        cfg = _minimal_v2_cfg()
        del cfg["labels"]["params"]
        import copy
        original = copy.deepcopy(cfg)
        normalize_ml_config(cfg)
        assert cfg == original

    def test_preserves_user_values(self):
        cfg = _minimal_v2_cfg()
        cfg["execution"] = {"transaction_cost_bps": 5.0}
        out = normalize_ml_config(cfg)
        assert out["execution"]["transaction_cost_bps"] == 5.0


# ---------------------------------------------------------------------------
# build_ml_experiment_spec
# ---------------------------------------------------------------------------


class TestBuildMLExperimentSpec:
    def test_builds_spec(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        spec = build_ml_experiment_spec(out)
        assert isinstance(spec, MLExperimentSpec)
        assert spec.name == "test_ml"
        assert spec.universe == ["SPY"]
        assert spec.start_date == "2020-01-01"
        assert spec.end_date == "2023-12-31"

    def test_feature_spec(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        spec = build_ml_experiment_spec(out)
        assert isinstance(spec.features, FeatureSpec)
        assert spec.features.ticker == "SPY"
        assert len(spec.features.entries) == 1
        assert spec.features.entries[0].name == "mom_20"

    def test_label_spec(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        spec = build_ml_experiment_spec(out)
        assert isinstance(spec.labels, LabelSpec)
        assert spec.labels.type == "forward_returns"
        assert spec.labels.params["horizon"] == 5

    def test_model_spec(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        spec = build_ml_experiment_spec(out)
        assert isinstance(spec.model, ModelSpec)
        assert spec.model.type == "RidgeRegression"

    def test_signal_spec(self):
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        spec = build_ml_experiment_spec(out)
        assert isinstance(spec.signal, SignalSpec)
        assert spec.signal.type == "sign"


# ---------------------------------------------------------------------------
# ml_experiment_hash
# ---------------------------------------------------------------------------


class TestMLExperimentHash:
    def _make_spec(self) -> MLExperimentSpec:
        cfg = _minimal_v2_cfg()
        out = normalize_ml_config(cfg)
        return build_ml_experiment_spec(out)

    def test_returns_12_char_string(self):
        spec = self._make_spec()
        h = ml_experiment_hash(spec)
        assert isinstance(h, str)
        assert len(h) == 12

    def test_deterministic(self):
        spec = self._make_spec()
        assert ml_experiment_hash(spec) == ml_experiment_hash(spec)

    def test_same_spec_same_hash(self):
        h1 = ml_experiment_hash(self._make_spec())
        h2 = ml_experiment_hash(self._make_spec())
        assert h1 == h2

    def test_different_model_params_different_hash(self):
        spec1 = self._make_spec()
        spec2 = self._make_spec()
        spec2.model.params["alpha"] = 99.0
        assert ml_experiment_hash(spec1) != ml_experiment_hash(spec2)

    def test_different_horizon_different_hash(self):
        spec1 = self._make_spec()
        spec2 = self._make_spec()
        spec2.labels.params["horizon"] = 20
        assert ml_experiment_hash(spec1) != ml_experiment_hash(spec2)

    def test_tags_excluded_from_hash(self):
        spec1 = self._make_spec()
        spec2 = self._make_spec()
        spec2.tags = ["research", "etf"]
        assert ml_experiment_hash(spec1) == ml_experiment_hash(spec2)

    def test_description_excluded_from_hash(self):
        spec1 = self._make_spec()
        spec2 = self._make_spec()
        spec2.description = "Different description"
        assert ml_experiment_hash(spec1) == ml_experiment_hash(spec2)

    def test_hex_chars_only(self):
        spec = self._make_spec()
        h = ml_experiment_hash(spec)
        assert all(c in "0123456789abcdef" for c in h)
