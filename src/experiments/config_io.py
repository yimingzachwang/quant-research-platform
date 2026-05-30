"""YAML/JSON config loading, validation, and normalization for D1 experiment configs.

Three operations, always in this order:
    raw_cfg  = load_config(path)       # file I/O only — no transformation
               validate_config(raw_cfg) # raises ValueError on structural violations
    norm_cfg = normalize_config(raw_cfg)# fills defaults, returns new dict

Keeping them separate allows each to be tested independently and lets the
orchestrator decide when/whether to normalize.

D1 config schema version: "1"
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_VERSION = "1"
_VALID_STRATEGY_TYPES = {"MomentumRotation", "EqualWeight", "BuyAndHold"}
_VALID_VALIDATION_TYPES = {"rolling", "expanding", "none"}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config from a YAML or JSON file.

    Returns the raw parsed dict with no defaults applied and no transformation.
    File format is detected by extension (.yaml/.yml → YAML, .json → JSON).

    Args:
        path: Path to the config file.

    Returns:
        Raw config dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or the content is not a mapping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open(encoding="utf-8") as f:
        if p.suffix in (".yaml", ".yml"):
            cfg = yaml.safe_load(f) or {}
        elif p.suffix == ".json":
            cfg = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {p.suffix!r}. Use .yaml, .yml, or .json"
            )

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must be a YAML/JSON mapping, got {type(cfg).__name__}: {p}"
        )

    return cfg


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate_config(cfg: dict[str, Any]) -> None:
    """Check required fields and structural correctness of a raw config.

    Operates on the raw (un-normalized) dict.  Raises ``ValueError`` with a
    descriptive message on the first violation found.  Does not modify ``cfg``.

    Args:
        cfg: Raw config dict from ``load_config()``.

    Raises:
        ValueError: On any structural violation.
    """
    # Schema version
    version = cfg.get("version")
    if version is not None and str(version) != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config schema version {version!r}. "
            f"Expected {_SCHEMA_VERSION!r}."
        )

    # name
    name = cfg.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        raise ValueError("Config requires a non-empty string 'name'.")

    # universe.tickers
    universe = cfg.get("universe")
    if not isinstance(universe, dict):
        raise ValueError("'universe' must be a mapping with a 'tickers' key.")
    tickers = universe.get("tickers")
    if (
        not tickers
        or not isinstance(tickers, list)
        or not all(isinstance(t, str) and t.strip() for t in tickers)
    ):
        raise ValueError("'universe.tickers' must be a non-empty list of non-empty strings.")

    # date_range
    date_range = cfg.get("date_range")
    if not isinstance(date_range, dict):
        raise ValueError("'date_range' must be a mapping with 'start' and 'end' keys.")
    start = date_range.get("start")
    end = date_range.get("end")
    if not start or not end:
        raise ValueError("'date_range.start' and 'date_range.end' are required.")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("'date_range.start' and 'date_range.end' must be ISO date strings.")
    if start >= end:
        raise ValueError(
            f"'date_range.start' ({start!r}) must be before 'date_range.end' ({end!r})."
        )

    # strategy
    strategy = cfg.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError("'strategy' must be a mapping.")
    stype = strategy.get("type")
    if not stype:
        raise ValueError("'strategy.type' is required.")
    if stype not in _VALID_STRATEGY_TYPES:
        raise ValueError(
            f"Unknown strategy type {stype!r}. "
            f"Available: {sorted(_VALID_STRATEGY_TYPES)}"
        )
    sparams = strategy.get("parameters")
    if sparams is not None and not isinstance(sparams, dict):
        raise ValueError("'strategy.parameters' must be a mapping.")

    # validation (optional section)
    validation = cfg.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            raise ValueError("'validation' must be a mapping.")
        vtype = validation.get("type", "none")
        if vtype not in _VALID_VALIDATION_TYPES:
            raise ValueError(
                f"Unknown validation type {vtype!r}. "
                f"Available: {sorted(_VALID_VALIDATION_TYPES)}"
            )
        vparams = validation.get("parameters", {})
        if vparams is not None and not isinstance(vparams, dict):
            raise ValueError("'validation.parameters' must be a mapping.")
        if vtype != "none" and isinstance(vparams, dict):
            for key in ("train_months", "test_months"):
                val = vparams.get(key)
                if val is not None and (not isinstance(val, int) or val <= 0):
                    raise ValueError(
                        f"'validation.parameters.{key}' must be a positive integer, "
                        f"got {val!r}."
                    )

    # execution (optional section)
    execution = cfg.get("execution")
    if execution is not None:
        if not isinstance(execution, dict):
            raise ValueError("'execution' must be a mapping.")
        cost = execution.get("transaction_cost_bps")
        if cost is not None and (not isinstance(cost, (int, float)) or cost < 0):
            raise ValueError(
                f"'execution.transaction_cost_bps' must be a non-negative number, "
                f"got {cost!r}."
            )

    # output (optional section)
    output = cfg.get("output")
    if output is not None and not isinstance(output, dict):
        raise ValueError("'output' must be a mapping.")


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical copy of cfg with all optional fields filled to defaults.

    Rules:
    - Never mutates the input dict.
    - Fills missing optional keys with defaults.
    - Merges nested dicts (validation, execution, output) preserving user values.
    - Semantically equivalent configs (same content, different key order or
      missing optionals) normalize to the same structure.
    - Does NOT fill strategy-specific parameter defaults — that is the
      strategy dataclass's responsibility.

    Args:
        cfg: Raw config dict, already validated by ``validate_config()``.

    Returns:
        New dict with all defaults applied.
    """
    out = copy.deepcopy(cfg)

    # version
    out.setdefault("version", _SCHEMA_VERSION)

    # metadata
    out.setdefault("description", "")
    if not isinstance(out.get("tags"), list):
        out["tags"] = list(out.get("tags", []))

    # strategy.parameters
    out["strategy"].setdefault("parameters", {})

    # validation
    if "validation" not in out or not isinstance(out["validation"], dict):
        out["validation"] = {"type": "none", "parameters": {}}
    else:
        out["validation"].setdefault("type", "none")
        out["validation"].setdefault("parameters", {})
        vtype = out["validation"]["type"]
        vparams = out["validation"]["parameters"]
        if vtype in ("rolling", "expanding"):
            vparams.setdefault("gap_days", 0)
            # step_months defaults to test_months if omitted
            if "step_months" not in vparams and "test_months" in vparams:
                vparams["step_months"] = vparams["test_months"]

    # execution
    if "execution" not in out or not isinstance(out["execution"], dict):
        out["execution"] = {"transaction_cost_bps": 0.0}
    else:
        out["execution"].setdefault("transaction_cost_bps", 0.0)
        # Ensure numeric type consistency
        out["execution"]["transaction_cost_bps"] = float(
            out["execution"]["transaction_cost_bps"]
        )

    # output
    _output_defaults: dict[str, Any] = {
        "base_dir": "results/experiments",
        "registry_path": "results/experiments/registry.json",
        "register": True,
        "save_plots": True,
    }
    if "output" not in out or not isinstance(out["output"], dict):
        out["output"] = dict(_output_defaults)
    else:
        for key, default in _output_defaults.items():
            out["output"].setdefault(key, default)

    return out
