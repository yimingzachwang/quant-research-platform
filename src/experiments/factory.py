"""Factory layer for Phase D1 declarative research configuration.

Pure functions that map normalized config dicts → typed Python objects.
No I/O, no data loading, no side effects.

Call order:
    norm_cfg = normalize_config(raw_cfg)
    strategy  = build_strategy(norm_cfg["strategy"])
    uni_spec  = build_universe_spec(norm_cfg["universe"], norm_cfg["date_range"])
    val_cfg   = build_validation_config(norm_cfg["validation"])
    spec      = build_experiment_spec(norm_cfg)
    splits    = build_validation_splits(val_cfg, price_index)  # needs real index
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.experiments.config import ExperimentSpec
from src.strategies.base import Strategy
from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.validation.splits import TimeSplit, expanding_time_splits, rolling_time_splits

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "MomentumRotation": MomentumRotationStrategy,
    "EqualWeight": EqualWeightStrategy,
    "BuyAndHold": BuyAndHoldStrategy,
}


# ---------------------------------------------------------------------------
# Data classes returned by factory functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseSpec:
    """Lightweight universe specification — no data loading."""

    tickers: tuple[str, ...]
    start_date: str
    end_date: str


@dataclass
class ValidationConfig:
    """Parsed validation section from a normalized config."""

    type: str  # "rolling", "expanding", or "none"
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_strategy(strategy_cfg: dict[str, Any]) -> Strategy:
    """Instantiate a strategy from the normalized ``strategy`` config block.

    Args:
        strategy_cfg: Normalized strategy sub-dict with ``type`` and
            ``parameters`` keys.

    Returns:
        Instantiated Strategy object.

    Raises:
        ValueError: If the strategy type is unknown.
    """
    stype = strategy_cfg["type"]
    if stype not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy type {stype!r}. "
            f"Available: {sorted(_STRATEGY_REGISTRY)}"
        )
    params = dict(strategy_cfg.get("parameters") or {})
    return _STRATEGY_REGISTRY[stype](**params)


def build_universe_spec(
    universe_cfg: dict[str, Any],
    date_range_cfg: dict[str, Any],
) -> UniverseSpec:
    """Build a UniverseSpec from the normalized universe and date_range blocks.

    Args:
        universe_cfg: Normalized ``universe`` sub-dict with ``tickers``.
        date_range_cfg: Normalized ``date_range`` sub-dict with ``start``/``end``.

    Returns:
        UniverseSpec (immutable, no data loaded).
    """
    tickers = tuple(universe_cfg["tickers"])
    return UniverseSpec(
        tickers=tickers,
        start_date=str(date_range_cfg["start"]),
        end_date=str(date_range_cfg["end"]),
    )


def build_validation_config(validation_cfg: dict[str, Any]) -> ValidationConfig:
    """Build a ValidationConfig from the normalized ``validation`` block.

    Args:
        validation_cfg: Normalized validation sub-dict.

    Returns:
        ValidationConfig dataclass.
    """
    return ValidationConfig(
        type=validation_cfg.get("type", "none"),
        parameters=dict(validation_cfg.get("parameters") or {}),
    )


def build_experiment_spec(cfg: dict[str, Any]) -> ExperimentSpec:
    """Build an ExperimentSpec from a fully normalized config dict.

    Args:
        cfg: Normalized config dict (output of ``normalize_config()``).

    Returns:
        ExperimentSpec suitable for hashing and registry storage.
    """
    strategy_cfg = cfg["strategy"]
    date_range = cfg["date_range"]
    rebalance_freq = strategy_cfg.get("parameters", {}).get("rebalance_freq", "ME")

    return ExperimentSpec(
        experiment_name=cfg["name"],
        strategy_name=strategy_cfg["type"],
        universe=list(cfg["universe"]["tickers"]),
        start_date=str(date_range["start"]),
        end_date=str(date_range["end"]),
        rebalance_frequency=rebalance_freq,
        parameters=dict(strategy_cfg.get("parameters") or {}),
        tags=list(cfg.get("tags") or []),
        description=cfg.get("description", ""),
    )


def build_validation_splits(
    validation_config: ValidationConfig,
    index: pd.DatetimeIndex,
) -> list[TimeSplit]:
    """Generate time splits from a ValidationConfig and a price index.

    Args:
        validation_config: Parsed validation configuration.
        index: DatetimeIndex of available price data.

    Returns:
        List of TimeSplit objects; empty list if validation type is ``"none"``.

    Raises:
        ValueError: If required parameters are missing for rolling/expanding.
    """
    vtype = validation_config.type
    params = validation_config.parameters

    if vtype == "none":
        return []

    if vtype == "rolling":
        train_months = params.get("train_months")
        test_months = params.get("test_months")
        if not train_months or not test_months:
            raise ValueError(
                "'validation.parameters' must include 'train_months' and "
                "'test_months' for rolling validation."
            )
        return rolling_time_splits(
            index=index,
            train_months=int(train_months),
            test_months=int(test_months),
            step_months=params.get("step_months"),
            gap_days=int(params.get("gap_days", 0)),
        )

    if vtype == "expanding":
        min_train_months = params.get("train_months")
        test_months = params.get("test_months")
        if not min_train_months or not test_months:
            raise ValueError(
                "'validation.parameters' must include 'train_months' and "
                "'test_months' for expanding validation."
            )
        return expanding_time_splits(
            index=index,
            min_train_months=int(min_train_months),
            test_months=int(test_months),
            step_months=params.get("step_months"),
            gap_days=int(params.get("gap_days", 0)),
        )

    raise ValueError(f"Unknown validation type {vtype!r}.")


def available_strategies() -> list[str]:
    """Return a sorted list of registered strategy type names."""
    return sorted(_STRATEGY_REGISTRY)
