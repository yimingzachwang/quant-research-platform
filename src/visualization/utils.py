"""Lightweight visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_figure(fig: plt.Figure, path: str | Path, close: bool = True) -> Path:
    """Save ``fig`` to ``path``, creating parent directories as needed.

    Args:
        fig: The matplotlib Figure to save.
        path: Destination file path.  Extension determines format (.png, .pdf, etc.).
        close: Close the figure after saving to release memory (default True).

    Returns:
        Resolved Path of the saved file.
    """
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    if close:
        plt.close(fig)
    return dest


def to_datetime_index(series: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Ensure the object has a DatetimeIndex.

    If the index is already a DatetimeIndex, returns the object unchanged.
    If the index is a RangeIndex but a 'timestamp' column exists, sets it as the
    index and drops timezone info for clean axis labels.
    """
    if isinstance(series.index, pd.DatetimeIndex):
        return series
    if isinstance(series, pd.DataFrame) and "timestamp" in series.columns:
        out = series.set_index("timestamp")
        out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
        return out
    return series


def validate_index_alignment(a: pd.Series, b: pd.Series, name_a: str = "a", name_b: str = "b") -> None:
    """Raise ValueError if two series do not share a common index."""
    if not a.index.equals(b.index):
        raise ValueError(
            f"Index mismatch between '{name_a}' and '{name_b}'. "
            "Align series before plotting."
        )


def pct_formatter(x: float, _: object) -> str:
    """Matplotlib tick formatter that renders a float as a percentage string."""
    return f"{x:.0%}"
