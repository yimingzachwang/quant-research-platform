"""Run a dataset profile through Data Agent V1 ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetUpdateEngine  # noqa: E402


def main() -> None:
    """Execute a dataset profile and print registered manifests."""

    parser = argparse.ArgumentParser(description="Update a configured dataset profile.")
    parser.add_argument("profile", help="Path to a dataset profile YAML file.")
    args = parser.parse_args()

    manifests = DatasetUpdateEngine(project_root=PROJECT_ROOT).run_profile(args.profile)
    payload = [manifest.model_dump(mode="json") for manifest in manifests]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
