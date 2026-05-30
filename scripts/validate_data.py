"""Compatibility wrapper for validating configured datasets."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import validate_data  # noqa: E402


def main() -> None:
    """Print a placeholder validation result for a dataset config."""
    validate_data()


if __name__ == "__main__":
    main()
