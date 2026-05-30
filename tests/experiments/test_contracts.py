"""Tests for src.experiments.contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.contracts import (
    ARTEFACT_VERSION,
    ARTEFACT_DIRS,
    CONFIG_ARTEFACTS,
    CORE_ARTEFACTS,
    D1_ARTEFACTS,
    REQUIRED_ARTEFACTS,
    check_artefact_dir,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_artefact_version_is_string():
    assert isinstance(ARTEFACT_VERSION, str)


def test_artefact_version_value():
    assert ARTEFACT_VERSION == "1"


def test_required_artefacts_are_tuple():
    assert isinstance(REQUIRED_ARTEFACTS, tuple)
    assert len(REQUIRED_ARTEFACTS) >= 1


def test_required_artefacts_contain_expected_files():
    assert "metadata.json" in REQUIRED_ARTEFACTS
    assert "metrics.json" in REQUIRED_ARTEFACTS


def test_constants_are_tuples():
    for const in (CORE_ARTEFACTS, CONFIG_ARTEFACTS, D1_ARTEFACTS, ARTEFACT_DIRS):
        assert isinstance(const, tuple)


# ---------------------------------------------------------------------------
# check_artefact_dir — advisory, never raises
# ---------------------------------------------------------------------------


def test_check_missing_directory_returns_violation():
    violations = check_artefact_dir("/tmp/does_not_exist_xyz_987")
    assert len(violations) == 1
    assert "does not exist" in violations[0].lower() or "directory" in violations[0].lower()


def test_check_empty_directory_returns_violations(tmp_path):
    violations = check_artefact_dir(tmp_path)
    # Both required artefacts are missing
    assert len(violations) == len(REQUIRED_ARTEFACTS)


def test_check_partial_directory_returns_one_violation(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"experiment_name": "test"}))
    violations = check_artefact_dir(tmp_path)
    # metrics.json still missing
    assert len(violations) == 1
    assert "metrics.json" in violations[0]


def test_check_complete_directory_returns_no_violations(tmp_path):
    for name in REQUIRED_ARTEFACTS:
        (tmp_path / name).write_text("{}")
    violations = check_artefact_dir(tmp_path)
    assert violations == []


def test_check_artefact_dir_accepts_str(tmp_path):
    for name in REQUIRED_ARTEFACTS:
        (tmp_path / name).write_text("{}")
    violations = check_artefact_dir(str(tmp_path))
    assert violations == []


def test_check_artefact_dir_never_raises(tmp_path):
    # Even with a completely empty path, should not raise
    result = check_artefact_dir("/nonexistent/deeply/nested/path")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Integration: ARTEFACT_VERSION in public __init__
# ---------------------------------------------------------------------------


def test_artefact_version_importable_from_package():
    from src.experiments import ARTEFACT_VERSION as av
    assert av == "1"


def test_check_artefact_dir_importable_from_package():
    from src.experiments import check_artefact_dir as cad
    assert callable(cad)


def test_required_artefacts_importable_from_package():
    from src.experiments import REQUIRED_ARTEFACTS as ra
    assert isinstance(ra, tuple)
