"""Safe JSON and Parquet loading utilities.

Every load function returns None (not raises) on missing or malformed files,
so retrievers can handle partial artefact sets gracefully without special-casing
every caller.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_json(path: Path | str) -> dict[str, Any] | list[Any] | None:
    """Load a JSON file; return None on missing file or parse error."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        logger.debug("load_json failed for %s: %s", p, exc)
        return None


def load_parquet(path: Path | str) -> pd.DataFrame | None:
    """Load a Parquet file as DataFrame; return None on missing or error."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.debug("load_parquet failed for %s: %s", p, exc)
        return None


def load_series_parquet(path: Path | str, column: str | None = None) -> pd.Series | None:
    """Load a Parquet file and return the first column (or named column) as Series."""
    df = load_parquet(path)
    if df is None or df.empty:
        return None
    if column is not None and column in df.columns:
        return df[column]
    return df.iloc[:, 0]


def dump_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    """Write obj to path as JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent, default=_json_default))


def _json_default(obj: Any) -> Any:
    """Fallback serializer for types not natively JSON-serializable."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
