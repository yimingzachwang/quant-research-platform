"""Tests for src/experiments/ml_factory.py — F3 ML factory layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.experiments.ml_config import (
    FeatureEntry,
    FeatureSpec,
    LabelSpec,
    ModelSpec,
    SignalSpec,
)
from src.experiments.ml_factory import (
    build_feature_fns,
    build_label_fn,
    build_ml_strategy,
    build_model,
    build_signal_fn,
)
from src.ml.contracts import PredictionSeries
from src.strategies.ml_strategy import MLStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(ticker: str = "SPY", n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({ticker: close}, index=idx)


# ---------------------------------------------------------------------------
# build_feature_fns
# ---------------------------------------------------------------------------


class TestBuildFeatureFns:
    def test_momentum_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("mom", "momentum", {"lookback": 20})],
        )
        fns = build_feature_fns(fs)
        assert "mom" in fns
        prices = _make_prices()
        result = fns["mom"](prices)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_rolling_volatility_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("vol", "rolling_volatility", {"window": 21})],
        )
        fns = build_feature_fns(fs)
        prices = _make_prices()
        result = fns["vol"](prices)
        assert isinstance(result, pd.Series)

    def test_rolling_zscore_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("zscore", "rolling_zscore", {"window": 20})],
        )
        fns = build_feature_fns(fs)
        prices = _make_prices()
        result = fns["zscore"](prices)
        assert isinstance(result, pd.Series)

    def test_sma_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("sma20", "sma", {"window": 20})],
        )
        fns = build_feature_fns(fs)
        prices = _make_prices()
        result = fns["sma20"](prices)
        assert isinstance(result, pd.Series)

    def test_ema_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("ema10", "ema", {"span": 10})],
        )
        fns = build_feature_fns(fs)
        prices = _make_prices()
        result = fns["ema10"](prices)
        assert isinstance(result, pd.Series)

    def test_compute_returns_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("ret", "compute_returns", {})],
        )
        fns = build_feature_fns(fs)
        prices = _make_prices()
        result = fns["ret"](prices)
        assert isinstance(result, pd.Series)

    def test_trend_strength_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("trend", "trend_strength", {"window": 20})],
        )
        fns = build_feature_fns(fs)
        prices = _make_prices()
        result = fns["trend"](prices)
        assert isinstance(result, pd.Series)

    def test_multiple_features(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[
                FeatureEntry("mom", "momentum", {"lookback": 20}),
                FeatureEntry("vol", "rolling_volatility", {"window": 21}),
                FeatureEntry("ret", "compute_returns", {}),
            ],
        )
        fns = build_feature_fns(fs)
        assert set(fns.keys()) == {"mom", "vol", "ret"}

    def test_closure_captures_params_correctly(self):
        # Two entries with different lookbacks — closure must not share params
        fs = FeatureSpec(
            ticker="SPY",
            entries=[
                FeatureEntry("mom_20", "momentum", {"lookback": 20}),
                FeatureEntry("mom_60", "momentum", {"lookback": 60}),
            ],
        )
        fns = build_feature_fns(fs)
        prices = _make_prices(n=130)
        r20 = fns["mom_20"](prices)
        r60 = fns["mom_60"](prices)
        # Different lookbacks → different NaN counts
        assert r20.isna().sum() != r60.isna().sum()

    def test_unknown_feature_type_raises(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("x", "unknown_type", {})],
        )
        with pytest.raises(ValueError, match="Unknown feature type"):
            build_feature_fns(fs)

    def test_empty_entries_returns_empty_dict(self):
        fs = FeatureSpec(ticker="SPY", entries=[])
        fns = build_feature_fns(fs)
        assert fns == {}

    def test_downside_volatility_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("dsvol", "downside_volatility", {"window": 21})],
        )
        fns = build_feature_fns(fs)
        result = fns["dsvol"](_make_prices(n=150))
        assert isinstance(result, pd.Series)

    def test_vol_of_vol_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("vov", "vol_of_vol", {"vol_window": 21, "meta_window": 63})],
        )
        fns = build_feature_fns(fs)
        result = fns["vov"](_make_prices(n=200))
        assert isinstance(result, pd.Series)

    def test_vol_percentile_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("vp", "vol_percentile", {"vol_window": 21, "lookback": 60})],
        )
        fns = build_feature_fns(fs)
        result = fns["vp"](_make_prices(n=200))
        assert isinstance(result, pd.Series)

    def test_bollinger_distance_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("bb", "bollinger_distance", {"window": 20})],
        )
        fns = build_feature_fns(fs)
        result = fns["bb"](_make_prices(n=150))
        assert isinstance(result, pd.Series)

    def test_rolling_skewness_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("sk", "rolling_skewness", {"window": 60})],
        )
        fns = build_feature_fns(fs)
        result = fns["sk"](_make_prices(n=200))
        assert isinstance(result, pd.Series)

    def test_rolling_autocorrelation_callable(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("ac", "rolling_autocorrelation", {"lag": 1, "window": 60})],
        )
        fns = build_feature_fns(fs)
        result = fns["ac"](_make_prices(n=200))
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# build_label_fn
# ---------------------------------------------------------------------------


class TestBuildLabelFn:
    def _fs(self) -> FeatureSpec:
        return FeatureSpec(ticker="SPY", entries=[])

    def test_forward_returns(self):
        ls = LabelSpec("forward_returns", {"horizon": 5})
        fn = build_label_fn(ls, self._fs())
        prices = _make_prices()
        result = fn(prices)
        assert isinstance(result, pd.Series)
        assert result.isna().sum() >= 5  # last 5 rows are NaN

    def test_binary_direction(self):
        ls = LabelSpec("binary_direction", {"horizon": 5})
        fn = build_label_fn(ls, self._fs())
        prices = _make_prices()
        result = fn(prices)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_volatility_target(self):
        ls = LabelSpec("volatility_target", {"horizon": 5})
        fn = build_label_fn(ls, self._fs())
        prices = _make_prices()
        result = fn(prices)
        assert isinstance(result, pd.Series)

    def test_panel_label_raises(self):
        ls = LabelSpec("ranking_target", {"horizon": 5})
        with pytest.raises(ValueError, match="panel experiment"):
            build_label_fn(ls, self._fs())

    def test_unknown_label_type_raises(self):
        ls = LabelSpec("mystery", {"horizon": 5})
        # Unknown type won't be in PANEL_LABEL_TYPES but will hit the fallback
        with pytest.raises(ValueError):
            build_label_fn(ls, self._fs())


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_linear_regression(self):
        from src.ml.models.linear import LinearRegressionModel
        model = build_model(ModelSpec("LinearRegression", {}))
        assert isinstance(model, LinearRegressionModel)

    def test_ridge_regression(self):
        from src.ml.models.linear import RidgeRegressionModel
        model = build_model(ModelSpec("RidgeRegression", {"alpha": 2.0}))
        assert isinstance(model, RidgeRegressionModel)

    def test_lasso_regression(self):
        from src.ml.models.linear import LassoRegressionModel
        model = build_model(ModelSpec("LassoRegression", {}))
        assert isinstance(model, LassoRegressionModel)

    def test_elasticnet_regression(self):
        from src.ml.models.linear import ElasticNetRegressionModel
        model = build_model(ModelSpec("ElasticNetRegression", {}))
        assert isinstance(model, ElasticNetRegressionModel)

    def test_logistic_regression(self):
        from src.ml.models.logistic import LogisticRegressionModel
        model = build_model(ModelSpec("LogisticRegression", {"C": 0.5}))
        assert isinstance(model, LogisticRegressionModel)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model(ModelSpec("NeuralNet", {}))


# ---------------------------------------------------------------------------
# build_signal_fn
# ---------------------------------------------------------------------------


class TestBuildSignalFn:
    def _make_preds(self) -> PredictionSeries:
        return PredictionSeries(
            values=pd.Series([0.1, -0.2, 0.3, 0.0], dtype="float64"),
            label_name="forward_returns",
            model_name="TestModel",
        )

    def test_sign_signal_returns_dataframe(self):
        fn = build_signal_fn(SignalSpec("sign", {}))
        result = fn(self._make_preds())
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1

    def test_sign_signal_values(self):
        fn = build_signal_fn(SignalSpec("sign", {}))
        result = fn(self._make_preds())
        vals = result.iloc[:, 0].values
        assert set(vals).issubset({-1.0, 0.0, 1.0})

    def test_threshold_signal_returns_dataframe(self):
        fn = build_signal_fn(SignalSpec("threshold", {"threshold": 0.0}))
        result = fn(self._make_preds())
        assert isinstance(result, pd.DataFrame)

    def test_threshold_signal_default_zero(self):
        fn = build_signal_fn(SignalSpec("threshold", {}))
        result = fn(self._make_preds())
        vals = result.iloc[:, 0].values
        assert set(vals).issubset({0.0, 1.0})

    def test_panel_signal_raises(self):
        for stype in ("top_n", "long_short", "normalize"):
            with pytest.raises(ValueError, match="panel experiment"):
                build_signal_fn(SignalSpec(stype, {"n": 2, "n_long": 2, "n_short": 2}))


# ---------------------------------------------------------------------------
# build_ml_strategy (integration)
# ---------------------------------------------------------------------------


class TestBuildMLStrategy:
    def _build(self, model_type: str = "RidgeRegression") -> MLStrategy:
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("mom", "momentum", {"lookback": 20})],
        )
        ls = LabelSpec("forward_returns", {"horizon": 5})
        ms = ModelSpec(model_type, {})
        ss = SignalSpec("sign", {})
        return build_ml_strategy(fs, ls, ms, ss)

    def test_returns_ml_strategy(self):
        strategy = self._build()
        assert isinstance(strategy, MLStrategy)

    def test_strategy_name_contains_model(self):
        strategy = self._build("LinearRegression")
        assert "LinearRegression" in strategy.name

    def test_strategy_can_fit_and_generate(self):
        strategy = self._build()
        prices = _make_prices(n=130)
        strategy.fit(prices)
        weights = strategy.generate_weights(prices)
        assert isinstance(weights, pd.DataFrame)
        assert len(weights) == len(prices)

    def test_panel_label_raises_at_build(self):
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("mom", "momentum", {"lookback": 20})],
        )
        ls = LabelSpec("ranking_target", {"horizon": 5})
        ms = ModelSpec("RidgeRegression", {})
        ss = SignalSpec("sign", {})
        with pytest.raises(ValueError, match="panel experiment"):
            build_ml_strategy(fs, ls, ms, ss)

    def test_two_instances_are_independent(self):
        # Each build_ml_strategy call should produce a fresh, unfitted model
        fs = FeatureSpec(
            ticker="SPY",
            entries=[FeatureEntry("mom", "momentum", {"lookback": 20})],
        )
        ls = LabelSpec("forward_returns", {"horizon": 5})
        ms = ModelSpec("RidgeRegression", {})
        ss = SignalSpec("sign", {})
        s1 = build_ml_strategy(fs, ls, ms, ss)
        s2 = build_ml_strategy(fs, ls, ms, ss)
        # s1 and s2 should be distinct objects
        assert s1 is not s2
