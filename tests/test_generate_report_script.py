"""Tests for scripts/generate_report.py — CLI argument parsing only.

Does not invoke generate_experiment_report; all filesystem interaction is mocked.
"""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "generate_report.py"

# Add project root so the script's sys.path manipulation is idempotent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location("generate_report", SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

PRESET_MAP = _mod.PRESET_MAP
main = _mod.main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(argv: list[str], mock_paths: MagicMock | None = None) -> MagicMock:
    """Run main() with given argv; return the generate_experiment_report mock."""
    if mock_paths is None:
        mock_paths = MagicMock(markdown=Path("a.md"), html=Path("a.html"), provenance=Path("a_prov.json"))
    with (
        patch.object(sys, "argv", ["generate_report.py"] + argv),
        patch.object(_mod, "generate_experiment_report", return_value=mock_paths) as mock_gen,
    ):
        main()
    return mock_gen


# ---------------------------------------------------------------------------
# PRESET_MAP contract
# ---------------------------------------------------------------------------

class TestPresetMap:
    def test_all_five_keys_present(self):
        assert set(PRESET_MAP) == {"standard", "canonical", "compact", "diagnostics", "audit"}

    def test_values_are_distinct_specs(self):
        specs = list(PRESET_MAP.values())
        assert len(set(specs)) == len(specs)

    def test_standard_is_default_export(self):
        from src.reporting.report_spec import STANDARD_REPORT
        assert PRESET_MAP["standard"] is STANDARD_REPORT

    def test_canonical_maps_to_canonical_showcase(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        assert PRESET_MAP["canonical"] is CANONICAL_SHOWCASE


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_default_preset_is_standard(self):
        from src.reporting.report_spec import STANDARD_REPORT
        mock = _run(["results/experiments/foo"])
        _, kwargs = mock.call_args
        assert kwargs["report_spec"] is STANDARD_REPORT

    def test_explicit_standard(self):
        from src.reporting.report_spec import STANDARD_REPORT
        mock = _run(["results/experiments/foo", "--preset", "standard"])
        assert mock.call_args[1]["report_spec"] is STANDARD_REPORT

    def test_canonical_preset(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        mock = _run(["results/experiments/foo", "--preset", "canonical"])
        assert mock.call_args[1]["report_spec"] is CANONICAL_SHOWCASE

    def test_compact_preset(self):
        from src.reporting.report_spec import COMPACT_REPORT
        mock = _run(["results/experiments/foo", "--preset", "compact"])
        assert mock.call_args[1]["report_spec"] is COMPACT_REPORT

    def test_diagnostics_preset(self):
        from src.reporting.report_spec import DIAGNOSTICS_REPORT
        mock = _run(["results/experiments/foo", "--preset", "diagnostics"])
        assert mock.call_args[1]["report_spec"] is DIAGNOSTICS_REPORT

    def test_audit_preset(self):
        from src.reporting.report_spec import AUDIT_REPORT
        mock = _run(["results/experiments/foo", "--preset", "audit"])
        assert mock.call_args[1]["report_spec"] is AUDIT_REPORT

    def test_invalid_preset_exits(self):
        with pytest.raises(SystemExit):
            _run(["results/experiments/foo", "--preset", "nonexistent"])

    def test_no_html_flag(self):
        mock = _run(["results/experiments/foo", "--no-html"])
        assert mock.call_args[1]["include_html"] is False

    def test_html_on_by_default(self):
        mock = _run(["results/experiments/foo"])
        assert mock.call_args[1]["include_html"] is True

    def test_output_dir_forwarded(self):
        mock = _run(["results/experiments/foo", "--output", "my_reports"])
        assert mock.call_args[1]["output_dir"] == "my_reports"

    def test_preset_and_no_html_compose(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        mock = _run(["results/experiments/foo", "--preset", "canonical", "--no-html"])
        assert mock.call_args[1]["report_spec"] is CANONICAL_SHOWCASE
        assert mock.call_args[1]["include_html"] is False
