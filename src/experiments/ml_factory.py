"""Factory functions for building ML experiment components from version "2" configs.

All functions are pure: no I/O, no data loading, no side effects.

Only single-asset (single-ticker) experiments are fully implemented.
Panel experiments (ranking_target label, top_n/long_short/normalize signals)
are schema-valid but raise ValueError — deferred to a future phase.

Dependency direction: this module depends on src.features.*, src.ml.*, and
src.strategies.  It does NOT import from src.experiments.orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from src.experiments.ml_config import (
    FeatureSpec,
    LabelSpec,
    ModelSpec,
    PortfolioConstructionSpec,
    SignalSpec,
    PANEL_LABEL_TYPES,
    PANEL_SIGNAL_TYPES,
)

if TYPE_CHECKING:
    from src.ml.contracts import PredictionSeries
    from src.ml.models.base import BaseMLModel
    from src.strategies.ml_strategy import MLStrategy
    from src.strategies.panel_ml_strategy import PanelMLStrategy


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------


def build_feature_fns(
    feature_spec: FeatureSpec,
) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Build feature callables from a FeatureSpec.

    Each callable takes the full price DataFrame (Date × Asset) and returns
    a pd.Series for the specified ticker.  Default-argument binding in closures
    captures ticker and params at construction time (avoids late-binding bugs).

    Args:
        feature_spec: FeatureSpec with ticker and list of FeatureEntry objects.

    Returns:
        Dict mapping feature name → callable(prices: pd.DataFrame) → pd.Series.
        Compatible with build_feature_matrix() from src.ml.feature_matrix.

    Raises:
        ValueError: If a feature type is unknown (should not happen after validation).
    """
    from src.features.momentum import momentum, risk_adjusted_momentum
    from src.features.returns import compute_returns
    from src.features.rolling import (
        bollinger_distance,
        rolling_autocorrelation,
        rolling_skewness,
        rolling_zscore,
    )
    from src.features.trend import breakout_strength, ema, sma, trend_persistence, trend_strength
    from src.features.volatility import (
        downside_volatility,
        drawdown_distance,
        rolling_volatility,
        vol_compression,
        vol_of_vol,
        vol_percentile,
    )
    from src.features.market import rolling_beta

    fns: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
    ticker = feature_spec.ticker

    for entry in feature_spec.entries:
        etype = entry.type
        params = entry.params
        name = entry.name

        if etype == "momentum":
            lookback = int(params["lookback"])
            fns[name] = lambda prices, t=ticker, w=lookback: momentum(prices[t], w)

        elif etype == "rolling_volatility":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: rolling_volatility(
                prices[t].pct_change(), w
            )

        elif etype == "rolling_zscore":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: rolling_zscore(prices[t], w)

        elif etype == "sma":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: sma(prices[t], w)

        elif etype == "ema":
            span = int(params["span"])
            fns[name] = lambda prices, t=ticker, s=span: ema(prices[t], s)

        elif etype == "compute_returns":
            fns[name] = lambda prices, t=ticker: compute_returns(prices[t])

        elif etype == "trend_strength":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: trend_strength(prices[t], w)

        elif etype == "downside_volatility":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: downside_volatility(
                prices[t].pct_change(), w
            )

        elif etype == "vol_of_vol":
            vol_win = int(params.get("vol_window", 21))
            meta_win = int(params.get("meta_window", 63))
            fns[name] = lambda prices, t=ticker, vw=vol_win, mw=meta_win: vol_of_vol(
                prices[t].pct_change(), vw, mw
            )

        elif etype == "vol_percentile":
            vol_win = int(params.get("vol_window", 21))
            lookback = int(params.get("lookback", 252))
            fns[name] = lambda prices, t=ticker, vw=vol_win, lb=lookback: vol_percentile(
                prices[t].pct_change(), vw, lb
            )

        elif etype == "bollinger_distance":
            window = int(params["window"])
            n_std = float(params.get("n_std", 2.0))
            fns[name] = lambda prices, t=ticker, w=window, ns=n_std: bollinger_distance(
                prices[t], w, ns
            )

        elif etype == "rolling_skewness":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: rolling_skewness(
                prices[t].pct_change(), w
            )

        elif etype == "rolling_autocorrelation":
            lag = int(params.get("lag", 1))
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, lg=lag, w=window: rolling_autocorrelation(
                prices[t].pct_change(), lg, w
            )

        elif etype == "trend_persistence":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: trend_persistence(prices[t], w)

        elif etype == "breakout_strength":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: breakout_strength(prices[t], w)

        elif etype == "drawdown_distance":
            window = int(params["window"])
            fns[name] = lambda prices, t=ticker, w=window: drawdown_distance(prices[t], w)

        elif etype == "vol_compression":
            short_win = int(params["short_window"])
            long_win = int(params["long_window"])
            fns[name] = lambda prices, t=ticker, sw=short_win, lw=long_win: vol_compression(
                prices[t].pct_change(), sw, lw
            )

        elif etype == "rolling_beta":
            window = int(params["window"])
            market_t = str(params.get("market_ticker", "SPY"))
            fns[name] = lambda prices, t=ticker, w=window, m=market_t: (
                rolling_beta(prices[t].pct_change(), prices[m].pct_change(), w)
                if m in prices.columns
                else pd.Series(float("nan"), index=prices.index, name=f"beta_{w}d")
            )

        elif etype == "risk_adjusted_momentum":
            mom_win = int(params["mom_window"])
            vol_win = int(params.get("vol_window", 63))
            fns[name] = lambda prices, t=ticker, mw=mom_win, vw=vol_win: risk_adjusted_momentum(
                prices[t], mw, vw
            )

        else:
            raise ValueError(f"Unknown feature type {etype!r}.")

    return fns


# ---------------------------------------------------------------------------
# Label function
# ---------------------------------------------------------------------------


def build_label_fn(
    label_spec: LabelSpec,
    feature_spec: FeatureSpec,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Build the label callable from a LabelSpec (single-asset).

    For single-asset labels, the callable takes prices (Date × Asset) and
    returns a pd.Series (forward target for the specified ticker).

    Args:
        label_spec: LabelSpec specifying label type and params.
        feature_spec: FeatureSpec — provides the ticker for single-asset labels.

    Returns:
        Callable(prices: pd.DataFrame) → pd.Series.

    Raises:
        ValueError: If label type is panel-only or unknown.
    """
    from src.ml.labels import (
        binary_direction_label,
        forward_returns,
        volatility_target,
    )

    ltype = label_spec.type
    if ltype in PANEL_LABEL_TYPES:
        raise ValueError(
            f"Label type {ltype!r} requires a panel experiment. "
            "Use build_panel_label_fn() for panel labels."
        )

    ticker = feature_spec.ticker
    horizon = int(label_spec.params["horizon"])

    if ltype == "forward_returns":
        return lambda prices, t=ticker, h=horizon: forward_returns(prices[t], h)

    if ltype == "binary_direction":
        return lambda prices, t=ticker, h=horizon: binary_direction_label(prices[t], h)

    if ltype == "volatility_target":
        return lambda prices, t=ticker, h=horizon: volatility_target(prices[t], h)

    raise ValueError(f"Unknown label type {ltype!r}.")


def build_panel_label_fn(
    label_spec: LabelSpec,
    tickers: list[str],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build the cross-sectional label callable for panel experiments.

    For ranking_target: at each date, ranks all assets by their horizon-period
    forward return and returns percentile ranks in (0, 1].

    Args:
        label_spec: LabelSpec with type "ranking_target" and horizon param.
        tickers: Universe tickers; used to select columns from prices.

    Returns:
        Callable(prices: pd.DataFrame) → pd.DataFrame (Date × Asset).
    """
    from src.ml.labels import ranking_target

    ltype = label_spec.type
    if ltype not in PANEL_LABEL_TYPES:
        raise ValueError(
            f"Label type {ltype!r} is not a panel label. "
            "Use build_label_fn() for single-asset labels."
        )

    horizon = int(label_spec.params["horizon"])

    if ltype == "ranking_target":
        def _panel_label_fn(prices: pd.DataFrame, h: int = horizon) -> pd.DataFrame:
            available = [t for t in tickers if t in prices.columns]
            return ranking_target(prices[available], h)
        return _panel_label_fn

    raise ValueError(f"Unknown panel label type {ltype!r}.")


def build_feature_fn_builder(
    feature_spec: FeatureSpec,
) -> Callable[[str], dict[str, Callable[[pd.DataFrame], pd.Series]]]:
    """Build a per-ticker feature function factory for panel experiments.

    Returns a callable fn(ticker: str) → {name: fn(prices) → pd.Series},
    applying the same feature types from feature_spec to any ticker.
    The feature_spec.ticker field is ignored; caller supplies the ticker.

    Args:
        feature_spec: FeatureSpec defining which feature types to compute.

    Returns:
        Callable(ticker: str) → feature_fns dict compatible with
        build_panel_feature_matrix().
    """
    # Capture entries (not the ticker) at construction time
    entries = feature_spec.entries

    def build_for_ticker(ticker: str) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
        from src.experiments.ml_config import FeatureEntry, FeatureSpec as FS
        ticker_spec = FS(ticker=ticker, entries=entries)
        return build_feature_fns(ticker_spec)

    return build_for_ticker


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(model_spec: ModelSpec) -> "BaseMLModel":
    """Instantiate an ML model from a ModelSpec.

    Args:
        model_spec: ModelSpec with model type and hyperparameters.

    Returns:
        Instantiated BaseMLModel (satisfies the Protocol).

    Raises:
        ValueError: If model type is unknown (should not happen after validation).
    """
    from src.ml.models.linear import (
        ElasticNetRegressionModel,
        LassoRegressionModel,
        LinearRegressionModel,
        RidgeRegressionModel,
    )
    from src.ml.models.logistic import LogisticRegressionModel

    mtype = model_spec.type
    params: dict[str, Any] = dict(model_spec.params)

    if mtype == "LinearRegression":
        return LinearRegressionModel()

    if mtype == "RidgeRegression":
        return RidgeRegressionModel(**params)

    if mtype == "LassoRegression":
        return LassoRegressionModel(**params)

    if mtype == "ElasticNetRegression":
        return ElasticNetRegressionModel(**params)

    if mtype == "LogisticRegression":
        return LogisticRegressionModel(**params)

    raise ValueError(
        f"Unknown model type {mtype!r}. "
        "Available: LinearRegression, RidgeRegression, LassoRegression, "
        "ElasticNetRegression, LogisticRegression."
    )


# ---------------------------------------------------------------------------
# Signal function
# ---------------------------------------------------------------------------


def build_signal_fn(
    signal_spec: SignalSpec,
    asset_name: str | None = None,
) -> "Callable[[PredictionSeries], pd.DataFrame]":
    """Build the signal-to-weights callable from a SignalSpec.

    Single-asset signals (sign, threshold) wrap the returned pd.Series in a
    one-column DataFrame so it is compatible with run_portfolio_backtest.
    The output column is named ``asset_name`` when provided so that
    run_portfolio_backtest's column intersection with the returns DataFrame
    (which uses ticker names) is non-empty.

    Panel signals (top_n, long_short, normalize) are schema-valid but not yet
    implemented — they raise ValueError at factory time.

    Args:
        signal_spec: SignalSpec with signal type and params.
        asset_name:  Column name for single-asset output DataFrame.  Should be
                     set to ``feature_spec.ticker`` so weights align with the
                     returns DataFrame used in run_portfolio_backtest.

    Returns:
        Callable(predictions: PredictionSeries) → pd.DataFrame (Date × Asset).

    Raises:
        ValueError: If signal type is panel-only (not yet implemented in F3).
    """
    from src.ml.signals.prediction import sign_signal, threshold_signal

    stype = signal_spec.type
    if stype in PANEL_SIGNAL_TYPES:
        raise ValueError(
            f"Signal type {stype!r} is a panel signal. "
            "Use build_panel_signal_fn() for panel experiments."
        )

    if stype == "sign":
        def _sign_fn(preds: "PredictionSeries", name: str | None = asset_name) -> pd.DataFrame:
            s = sign_signal(preds)
            if name is not None:
                s = s.rename(name)
            return s.to_frame()
        return _sign_fn

    if stype == "threshold":
        threshold = float(signal_spec.params.get("threshold", 0.0))

        def _threshold_fn(
            preds: "PredictionSeries",
            t: float = threshold,
            name: str | None = asset_name,
        ) -> pd.DataFrame:
            s = threshold_signal(preds, threshold=t)
            if name is not None:
                s = s.rename(name)
            return s.to_frame()

        return _threshold_fn

    raise ValueError(f"Unknown signal type {stype!r}.")


# ---------------------------------------------------------------------------
# Composite: full MLStrategy
# ---------------------------------------------------------------------------


def build_panel_signal_fn(
    signal_spec: SignalSpec,
) -> "Callable[[PredictionSeries], pd.DataFrame]":
    """Build the panel signal-to-weights callable from a SignalSpec.

    Handles top_n, long_short, and normalize signal types.  Each function
    receives a PredictionSeries whose values is a pd.DataFrame (Date × Asset).

    Args:
        signal_spec: SignalSpec with panel signal type and params.

    Returns:
        Callable(predictions: PredictionSeries) → pd.DataFrame (Date × Asset).

    Raises:
        ValueError: If signal type is single-asset or unknown.
    """
    from src.ml.signals.prediction import long_short_weights, normalize_to_weights, top_n_weights

    stype = signal_spec.type
    if stype not in PANEL_SIGNAL_TYPES:
        raise ValueError(
            f"Signal type {stype!r} is not a panel signal. "
            "Use build_signal_fn() for single-asset signals."
        )

    if stype == "top_n":
        n = int(signal_spec.params["n"])
        return lambda preds, _n=n: top_n_weights(preds, _n)

    if stype == "long_short":
        n_long = int(signal_spec.params["n_long"])
        n_short = int(signal_spec.params["n_short"])
        return lambda preds, nl=n_long, ns=n_short: long_short_weights(preds, nl, ns)

    if stype == "normalize":
        return normalize_to_weights

    raise ValueError(f"Unknown panel signal type {stype!r}.")


def _build_panel_signal_fn_with_policy(
    signal_spec: SignalSpec,
    pc_spec: PortfolioConstructionSpec,
) -> Callable:
    """Build a panel signal function that applies a configurable weighting policy.

    Separates selection (ranking → top-N mask) from weighting (policy applied to
    mask and raw scores).  Only implemented for "top_n" signal type; all other
    panel signal types fall back to build_panel_signal_fn() (equal-weight baseline).

    Args:
        signal_spec: Panel signal specification (type and params).
        pc_spec: Portfolio construction specification (weighting policy).

    Returns:
        Callable(PredictionSeries) → pd.DataFrame (Date × Asset weights).
    """
    from src.portfolio.ranking import rank_assets, select_top_n
    from src.portfolio.weighting_policy import apply_weighting_policy

    stype = signal_spec.type
    weighting = pc_spec.weighting

    if stype == "top_n":
        n = int(signal_spec.params["n"])
        scheme = weighting.scheme
        pred_norm = weighting.prediction_normalization
        temperature = weighting.temperature

        def _top_n_with_policy(
            preds: "PredictionSeries",
            _n: int = n,
            _scheme: str = scheme,
            _norm: str = pred_norm,
            _temp: "float | None" = temperature,
        ) -> pd.DataFrame:
            ranks = rank_assets(preds.values, ascending=False)
            mask = select_top_n(ranks, n=_n)
            return apply_weighting_policy(preds.values, mask, _scheme, _norm, _temp)

        return _top_n_with_policy

    # long_short and normalize: weighting policy deferred; fall back to equal-weight baseline
    return build_panel_signal_fn(signal_spec)


def build_panel_ml_strategy(
    feature_spec: FeatureSpec,
    label_spec: LabelSpec,
    model_spec: ModelSpec,
    signal_spec: SignalSpec,
    tickers: list[str],
    portfolio_construction: PortfolioConstructionSpec | None = None,
) -> "PanelMLStrategy":
    """Assemble a PanelMLStrategy from the four spec objects + universe tickers.

    Pure function: builds callables from specs, then constructs PanelMLStrategy.
    The feature_spec.ticker field is used only as a schema placeholder and is
    ignored at runtime — features are built for each ticker in tickers.

    Args:
        feature_spec: Feature construction specification (ticker field ignored).
        label_spec: Label specification (must be a panel label type).
        model_spec: Model type and hyperparameters.
        signal_spec: Signal/weight conversion specification (must be panel type).
        tickers: Ordered universe tickers to build features and labels for.

    Returns:
        PanelMLStrategy ready for use with run_strategy / run_walk_forward_validation.
    """
    from src.strategies.panel_ml_strategy import PanelMLStrategy

    feature_fn_builder = build_feature_fn_builder(feature_spec)
    label_fn = build_panel_label_fn(label_spec, tickers)
    model = build_model(model_spec)

    if (
        portfolio_construction is not None
        and not portfolio_construction.weighting.is_default()
    ):
        signal_fn = _build_panel_signal_fn_with_policy(signal_spec, portfolio_construction)
    else:
        signal_fn = build_panel_signal_fn(signal_spec)

    horizon = int(label_spec.params["horizon"])

    return PanelMLStrategy(
        model=model,
        tickers=tickers,
        feature_fn_builder=feature_fn_builder,
        label_fn=label_fn,
        horizon=horizon,
        signal_fn=signal_fn,
        label_name=label_spec.type,
    )


def build_ml_strategy(
    feature_spec: FeatureSpec,
    label_spec: LabelSpec,
    model_spec: ModelSpec,
    signal_spec: SignalSpec,
) -> "MLStrategy":
    """Assemble an MLStrategy from the four spec objects.

    Pure function: builds callables from specs, then constructs MLStrategy.
    No data loading, no I/O.

    Args:
        feature_spec: Feature construction specification.
        label_spec: Label generation specification.
        model_spec: Model type and hyperparameters.
        signal_spec: Signal/weight conversion specification.

    Returns:
        MLStrategy ready for use with run_strategy / run_walk_forward_validation.

    Raises:
        ValueError: If any spec type is panel-only (not yet implemented).
    """
    from src.strategies.ml_strategy import MLStrategy

    feature_fns = build_feature_fns(feature_spec)
    label_fn = build_label_fn(label_spec, feature_spec)
    model = build_model(model_spec)
    signal_fn = build_signal_fn(signal_spec, asset_name=feature_spec.ticker)
    horizon = int(label_spec.params["horizon"])

    return MLStrategy(
        model=model,
        feature_fns=feature_fns,
        label_fn=label_fn,
        horizon=horizon,
        signal_fn=signal_fn,
        label_name=label_spec.type,
    )
