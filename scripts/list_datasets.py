"""List registered datasets from the local manifest registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import DatasetManifest  # noqa: E402
from src.data.registry import DatasetRegistry  # noqa: E402


def format_dataset_table(manifests: list[DatasetManifest]) -> str:
    """Return a deterministic terminal table for registered datasets."""

    if not manifests:
        return "No datasets registered."

    headers = ("DATASET ID", "SOURCE", "FREQ", "ROWS", "SYMBOL")
    rows = [
        (
            manifest.dataset_id,
            manifest.source,
            manifest.frequency,
            "" if manifest.row_count is None else str(manifest.row_count),
            manifest.symbol,
        )
        for manifest in manifests
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
    separator = "-" * len(header)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows
    ]
    return "\n".join([header, separator, *body])


def load_filtered_manifests(
    *,
    registry_path: Path | str,
    symbol: str | None = None,
    source: str | None = None,
) -> list[DatasetManifest]:
    """Load manifests with optional exact CLI filters."""

    registry = DatasetRegistry(registry_path)
    manifests = registry.find(symbol=symbol, source=source)
    return sorted(
        manifests,
        key=lambda manifest: (
            manifest.dataset_id,
            manifest.source,
            manifest.frequency,
            manifest.symbol,
        ),
    )


def main() -> None:
    """Print a summary of registered datasets."""

    parser = argparse.ArgumentParser(description="List registered datasets.")
    parser.add_argument("--symbol", help="Filter by exact symbol.")
    parser.add_argument("--source", help="Filter by exact source.")
    parser.add_argument(
        "--registry",
        default=PROJECT_ROOT / "data" / "external" / "registry" / "datasets.json",
        help="Path to the dataset registry JSON file.",
    )
    args = parser.parse_args()

    manifests = load_filtered_manifests(
        registry_path=args.registry,
        symbol=args.symbol,
        source=args.source,
    )
    print(format_dataset_table(manifests))


if __name__ == "__main__":
    main()
