"""Lightweight tests for scripts/demo_ai_orchestration.py.

These tests deliberately avoid:
  - calling any live LLM (no network, no API keys)
  - requiring LM Studio
  - requiring full experiment-result artefacts
  - running any experiment

They verify the demo's pure, governance-critical helpers: argument parsing,
the artefact preflight, the stub draft response schema, and — most importantly
— that YAML rendering is blocked before approval and only succeeds after it.

The render tests need only the base config YAML (configs/experiments/
canonical_ml_showcase.yaml), which is part of the repository.
"""

from __future__ import annotations

import pytest
from scripts import demo_ai_orchestration as demo
from src.orchestration.config_generation.draft_schema import (
    DraftChange,
    ExperimentDraft,
    compute_draft_hash,
)
from src.orchestration.config_generation.draft_validator import approve_draft
from src.orchestration.config_generation.yaml_renderer import render_to_yaml

BASE_EXPERIMENT = "canonical_ml_showcase"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults_to_stub():
    args = demo.parse_args([])
    assert args.provider == "stub"
    assert args.experiment == BASE_EXPERIMENT
    assert args.base_url is None
    assert args.model is None
    assert args.dry_run is False
    assert args.transcript is False
    # Execution is opt-in and off by default.
    assert args.execute_approved is False
    assert args.yes is False
    assert args.execution_preset == "canonical"
    assert args.no_report is False


def test_parse_args_transcript_flag():
    args = demo.parse_args(["--transcript"])
    assert args.transcript is True


def test_parse_args_execution_flags():
    args = demo.parse_args(
        ["--execute-approved", "--yes", "--execution-preset", "compact", "--no-report"]
    )
    assert args.execute_approved is True
    assert args.yes is True
    assert args.execution_preset == "compact"
    assert args.no_report is True


def test_parse_args_lm_studio_shape():
    args = demo.parse_args(
        [
            "--provider",
            "openai",
            "--model",
            "qwen2.5-7b-instruct",
            "--base-url",
            "http://127.0.0.1:1234/v1",
        ]
    )
    assert args.provider == "openai"
    assert args.model == "qwen2.5-7b-instruct"
    assert args.base_url == "http://127.0.0.1:1234/v1"


def test_parse_args_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        demo.parse_args(["--provider", "lmstudio"])


# ---------------------------------------------------------------------------
# Governed execution gate
# ---------------------------------------------------------------------------


def _boom(_prompt):  # input_fn that must never be called
    raise AssertionError("input() should not be called")


def test_default_does_not_authorise_execution():
    args = demo.parse_args([])
    assert demo.execution_authorised(args, input_fn=_boom) is False


def test_yes_alone_does_not_authorise():
    # --yes without --execute-approved must do nothing.
    args = demo.parse_args(["--yes"])
    assert demo.execution_authorised(args, input_fn=_boom) is False


def test_execute_approved_requires_typed_run():
    args = demo.parse_args(["--execute-approved"])
    assert demo.execution_authorised(args, input_fn=lambda _p: "RUN") is True
    assert demo.execution_authorised(args, input_fn=lambda _p: "  RUN\n") is True


def test_wrong_confirmation_cancels_execution():
    args = demo.parse_args(["--execute-approved"])
    assert demo.execution_authorised(args, input_fn=lambda _p: "no") is False
    assert demo.execution_authorised(args, input_fn=lambda _p: "run") is False  # case-sensitive
    assert demo.execution_authorised(args, input_fn=lambda _p: "") is False


def test_execute_approved_with_yes_skips_prompt():
    args = demo.parse_args(["--execute-approved", "--yes"])
    # input_fn must not be consulted when --yes is present.
    assert demo.execution_authorised(args, input_fn=_boom) is True


# ---------------------------------------------------------------------------
# Artefact preflight
# ---------------------------------------------------------------------------


def test_check_artefacts_reports_missing(tmp_path):
    missing = demo.check_artefacts(
        "does_not_exist",
        results_root=tmp_path,
        configs_base=tmp_path,
    )
    # Both required result artefacts and the config should be reported missing.
    assert any("metadata.json" in m for m in missing)
    assert any("metrics.json" in m for m in missing)
    assert any(".yaml" in m for m in missing)


def test_check_artefacts_ready(tmp_path):
    root = tmp_path / "results" / BASE_EXPERIMENT
    root.mkdir(parents=True)
    (root / "metadata.json").write_text("{}")
    (root / "metrics.json").write_text("{}")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / f"{BASE_EXPERIMENT}.yaml").write_text("version: '2'\n")

    missing = demo.check_artefacts(
        BASE_EXPERIMENT,
        results_root=tmp_path / "results",
        configs_base=configs,
    )
    assert missing == []


# ---------------------------------------------------------------------------
# Stub draft response
# ---------------------------------------------------------------------------


def test_stub_draft_json_is_valid_schema():
    import json

    payload = json.loads(demo.stub_draft_json(BASE_EXPERIMENT))
    assert payload["proposed_name"] == f"{BASE_EXPERIMENT}_v2"
    assert isinstance(payload["changes"], list) and payload["changes"]
    change = payload["changes"][0]
    assert change["section"] == "model"
    assert change["field"] == "params.alpha"
    assert "rationale" in change


# ---------------------------------------------------------------------------
# Governance: render is blocked before approval, allowed after
# ---------------------------------------------------------------------------


def _make_draft(approved: bool = False) -> ExperimentDraft:
    changes = [
        DraftChange(
            section="model",
            field="params.alpha",
            current_value=0.5,
            proposed_value=1.0,
            rationale="Stronger L2 regularisation for validation robustness.",
        )
    ]
    draft = ExperimentDraft(
        draft_id="demo-test-draft",
        draft_hash=compute_draft_hash(BASE_EXPERIMENT, f"{BASE_EXPERIMENT}_v2", changes),
        base_experiment=BASE_EXPERIMENT,
        source_proposal_hash="proposalhash123",
        proposed_name=f"{BASE_EXPERIMENT}_v2",
        changes=changes,
        generated_at="2026-05-30T00:00:00+00:00",
    )
    if approved:
        return approve_draft(draft)
    return draft


def test_render_blocked_before_approval():
    draft = _make_draft(approved=False)
    with pytest.raises(ValueError):
        render_to_yaml(draft, dry_run=True)


def test_render_after_approval_contains_provenance():
    approved = _make_draft(approved=True)
    assert approved.approved is True
    yaml_str = render_to_yaml(approved, dry_run=True)
    # Provenance: the rendered YAML header embeds the draft hash.
    assert approved.draft_hash in yaml_str
    assert f"{BASE_EXPERIMENT}_v2" in yaml_str


# ---------------------------------------------------------------------------
# Governance: the demo never wires in experiment execution
# ---------------------------------------------------------------------------


def test_demo_does_not_invoke_experiment_execution():
    import inspect

    src = inspect.getsource(demo)
    # The demo must not shell out or call the quant engine execution path.
    # (The run_from_config.py path may appear only as documentation text telling
    # the researcher how to generate artefacts or run the config manually.)
    assert "subprocess" not in src
    assert "os.system" not in src
    assert "run_and_report" not in src
    assert "run_experiment_from_config" not in src
    assert "src.experiments.orchestrator" not in src
    # No execution function from the runner is imported.
    assert not hasattr(demo, "run_and_report")
    assert not hasattr(demo, "run_experiment_from_config")
