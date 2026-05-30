"""Tests for Phase 4 — Research Session Layer.

All tests use tmp_path for isolation.  No quant engine, no evolution chain,
no external artefacts are loaded.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.orchestration.session.session_manager import (
    create_session,
    load_session,
    record_event,
    summarize_session,
    update_session_status,
)
from src.orchestration.session.session_schema import (
    ResearchSession,
    SessionEvent,
    SessionEventType,
    SessionStatus,
)
from src.orchestration.utils.filesystem import (
    list_session_ids,
    session_json_path,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_session(tmp_path: Path, **kwargs) -> ResearchSession:
    defaults = dict(
        root_experiment="canonical_ml_showcase",
        research_goal="Explore regularization",
        sessions_base=tmp_path,
    )
    defaults.update(kwargs)
    return create_session(**defaults)


# ---------------------------------------------------------------------------
# SessionStatus constants
# ---------------------------------------------------------------------------


class TestSessionStatus:
    def test_active_constant(self):
        assert SessionStatus.ACTIVE == "active"

    def test_paused_constant(self):
        assert SessionStatus.PAUSED == "paused"

    def test_complete_constant(self):
        assert SessionStatus.COMPLETE == "complete"


# ---------------------------------------------------------------------------
# SessionEventType constants
# ---------------------------------------------------------------------------


class TestSessionEventType:
    def test_all_seven_types_defined(self):
        expected = {
            "REVIEW_GENERATED",
            "ITERATION_PROPOSAL_GENERATED",
            "DRAFT_GENERATED",
            "DRAFT_VALIDATED",
            "DRAFT_APPROVED",
            "YAML_RENDERED",
            "EXPERIMENT_LINKED",
        }
        actual = {v for k, v in vars(SessionEventType).items() if not k.startswith("_")}
        assert actual == expected


# ---------------------------------------------------------------------------
# SessionEvent schema
# ---------------------------------------------------------------------------


class TestSessionEvent:
    def test_to_dict_includes_all_fields(self):
        ev = SessionEvent(
            event_id="abc",
            event_type=SessionEventType.REVIEW_GENERATED,
            timestamp="2026-05-29T00:00:00+00:00",
            experiment_name="exp_a",
            data={"provider": "anthropic"},
        )
        d = ev.to_dict()
        assert d["event_id"] == "abc"
        assert d["event_type"] == "REVIEW_GENERATED"
        assert d["timestamp"] == "2026-05-29T00:00:00+00:00"
        assert d["experiment_name"] == "exp_a"
        assert d["data"] == {"provider": "anthropic"}

    def test_to_dict_is_json_serialisable(self):
        ev = SessionEvent(
            event_id="x",
            event_type=SessionEventType.DRAFT_GENERATED,
            timestamp="2026-05-29T00:00:00+00:00",
            experiment_name="exp_a",
            data={"draft_id": "123", "draft_hash": "abc", "proposed_name": "exp_a_v2"},
        )
        json.dumps(ev.to_dict())  # must not raise

    def test_data_defaults_to_empty_dict(self):
        ev = SessionEvent(
            event_id="x",
            event_type=SessionEventType.REVIEW_GENERATED,
            timestamp="t",
            experiment_name="exp",
        )
        assert ev.data == {}


# ---------------------------------------------------------------------------
# ResearchSession schema
# ---------------------------------------------------------------------------


class TestResearchSessionSchema:
    def test_to_dict_includes_all_fields(self):
        s = ResearchSession(
            session_id="sid",
            research_goal="test goal",
            root_experiment="exp_a",
            active_experiment="exp_a",
            status=SessionStatus.ACTIVE,
            created_at="2026-05-29T00:00:00+00:00",
            updated_at="2026-05-29T00:00:00+00:00",
        )
        d = s.to_dict()
        for key in [
            "session_id", "research_goal", "root_experiment",
            "active_experiment", "status", "created_at", "updated_at",
            "events", "active_draft_id",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_default_events_empty(self):
        s = ResearchSession(
            session_id="x", research_goal="g", root_experiment="e",
            active_experiment="e", status="active",
            created_at="t", updated_at="t",
        )
        assert s.events == []

    def test_default_active_draft_id_none(self):
        s = ResearchSession(
            session_id="x", research_goal="g", root_experiment="e",
            active_experiment="e", status="active",
            created_at="t", updated_at="t",
        )
        assert s.active_draft_id is None

    def test_to_dict_is_json_serialisable(self):
        s = ResearchSession(
            session_id="x", research_goal="g", root_experiment="e",
            active_experiment="e", status="active",
            created_at="t", updated_at="t",
        )
        json.dumps(s.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_research_session(self, tmp_path):
        s = _make_session(tmp_path)
        assert isinstance(s, ResearchSession)

    def test_status_is_active(self, tmp_path):
        s = _make_session(tmp_path)
        assert s.status == SessionStatus.ACTIVE

    def test_active_experiment_equals_root(self, tmp_path):
        s = _make_session(tmp_path)
        assert s.active_experiment == s.root_experiment

    def test_events_empty_on_creation(self, tmp_path):
        s = _make_session(tmp_path)
        assert s.events == []

    def test_active_draft_id_none_on_creation(self, tmp_path):
        s = _make_session(tmp_path)
        assert s.active_draft_id is None

    def test_generates_session_id_when_absent(self, tmp_path):
        s = _make_session(tmp_path)
        assert s.session_id
        uuid.UUID(s.session_id)  # valid UUID

    def test_accepts_explicit_session_id(self, tmp_path):
        sid = "my-custom-session-id"
        s = create_session("exp_a", "goal", session_id=sid, sessions_base=tmp_path)
        assert s.session_id == sid

    def test_creates_json_on_disk(self, tmp_path):
        s = _make_session(tmp_path)
        path = session_json_path(s.session_id, tmp_path)
        assert path.exists()

    def test_json_is_valid(self, tmp_path):
        s = _make_session(tmp_path)
        path = session_json_path(s.session_id, tmp_path)
        data = json.loads(path.read_text())
        assert data["session_id"] == s.session_id
        assert data["research_goal"] == "Explore regularization"

    def test_timestamps_set(self, tmp_path):
        s = _make_session(tmp_path)
        assert s.created_at
        assert s.updated_at
        assert "T" in s.created_at  # ISO format


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------


class TestLoadSession:
    def test_loads_existing_session(self, tmp_path):
        created = _make_session(tmp_path)
        loaded = load_session(created.session_id, tmp_path)
        assert loaded is not None
        assert loaded.session_id == created.session_id
        assert loaded.research_goal == created.research_goal

    def test_returns_none_for_missing_session(self, tmp_path):
        result = load_session("nonexistent-session-id", tmp_path)
        assert result is None

    def test_rehydrates_events(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         data={"provider": "stub"}, sessions_base=tmp_path)
        loaded = load_session(s.session_id, tmp_path)
        assert loaded is not None
        assert len(loaded.events) == 1
        assert isinstance(loaded.events[0], SessionEvent)
        assert loaded.events[0].event_type == SessionEventType.REVIEW_GENERATED

    def test_rehydrates_active_draft_id(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "exp_a",
                         data={"draft_id": "d123", "draft_hash": "h", "proposed_name": "v2"},
                         sessions_base=tmp_path)
        loaded = load_session(s.session_id, tmp_path)
        assert loaded is not None
        assert loaded.active_draft_id == "d123"

    def test_rehydrates_active_experiment(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.EXPERIMENT_LINKED, "exp_a",
                         data={"new_experiment": "exp_a_v2"},
                         sessions_base=tmp_path)
        loaded = load_session(s.session_id, tmp_path)
        assert loaded is not None
        assert loaded.active_experiment == "exp_a_v2"


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------


class TestRecordEvent:
    def test_appends_event(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        assert len(s.events) == 1

    def test_event_has_correct_type(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.ITERATION_PROPOSAL_GENERATED, "exp_a",
                         data={"context_hash": "c", "research_focus": "r"},
                         sessions_base=tmp_path)
        assert s.events[0].event_type == SessionEventType.ITERATION_PROPOSAL_GENERATED

    def test_event_has_correct_experiment(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "canonical_ml_showcase",
                         sessions_base=tmp_path)
        assert s.events[0].experiment_name == "canonical_ml_showcase"

    def test_event_data_stored(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         data={"provider": "anthropic"}, sessions_base=tmp_path)
        assert s.events[0].data == {"provider": "anthropic"}

    def test_updates_updated_at(self, tmp_path):
        s = _make_session(tmp_path)
        original_ts = s.updated_at
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        assert s.updated_at >= original_ts

    def test_multiple_events_appended_in_order(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.ITERATION_PROPOSAL_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        assert len(s.events) == 2
        assert s.events[0].event_type == SessionEventType.REVIEW_GENERATED
        assert s.events[1].event_type == SessionEventType.ITERATION_PROPOSAL_GENERATED

    def test_draft_generated_sets_active_draft_id(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "exp_a",
                         data={"draft_id": "d-001", "draft_hash": "h", "proposed_name": "v2"},
                         sessions_base=tmp_path)
        assert s.active_draft_id == "d-001"

    def test_yaml_rendered_clears_active_draft_id(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "exp_a",
                         data={"draft_id": "d-001", "draft_hash": "h", "proposed_name": "v2"},
                         sessions_base=tmp_path)
        assert s.active_draft_id == "d-001"
        s = record_event(s, SessionEventType.YAML_RENDERED, "exp_a",
                         data={"draft_id": "d-001", "config_path": "configs/experiments/v2.yaml"},
                         sessions_base=tmp_path)
        assert s.active_draft_id is None

    def test_experiment_linked_updates_active_experiment(self, tmp_path):
        s = _make_session(tmp_path)
        assert s.active_experiment == "canonical_ml_showcase"
        s = record_event(s, SessionEventType.EXPERIMENT_LINKED, "canonical_ml_showcase",
                         data={"new_experiment": "canonical_ml_showcase_v2"},
                         sessions_base=tmp_path)
        assert s.active_experiment == "canonical_ml_showcase_v2"

    def test_experiment_linked_without_new_experiment_key_is_noop(self, tmp_path):
        s = _make_session(tmp_path)
        original = s.active_experiment
        s = record_event(s, SessionEventType.EXPERIMENT_LINKED, "exp_a",
                         data={}, sessions_base=tmp_path)
        assert s.active_experiment == original

    def test_event_id_is_unique(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        ids = [e.event_id for e in s.events]
        assert len(set(ids)) == 2

    def test_persists_after_event(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        loaded = load_session(s.session_id, tmp_path)
        assert loaded is not None
        assert len(loaded.events) == 1

    def test_none_data_stored_as_empty_dict(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         data=None, sessions_base=tmp_path)
        assert s.events[0].data == {}


# ---------------------------------------------------------------------------
# update_session_status
# ---------------------------------------------------------------------------


class TestUpdateSessionStatus:
    def test_sets_status_to_paused(self, tmp_path):
        s = _make_session(tmp_path)
        s = update_session_status(s, SessionStatus.PAUSED, tmp_path)
        assert s.status == SessionStatus.PAUSED

    def test_sets_status_to_complete(self, tmp_path):
        s = _make_session(tmp_path)
        s = update_session_status(s, SessionStatus.COMPLETE, tmp_path)
        assert s.status == SessionStatus.COMPLETE

    def test_no_transition_guard(self, tmp_path):
        s = _make_session(tmp_path)
        s = update_session_status(s, SessionStatus.COMPLETE, tmp_path)
        # Can move back to active — no guard enforced
        s = update_session_status(s, SessionStatus.ACTIVE, tmp_path)
        assert s.status == SessionStatus.ACTIVE

    def test_updates_updated_at(self, tmp_path):
        s = _make_session(tmp_path)
        ts_before = s.updated_at
        s = update_session_status(s, SessionStatus.PAUSED, tmp_path)
        assert s.updated_at >= ts_before

    def test_persists_status(self, tmp_path):
        s = _make_session(tmp_path)
        s = update_session_status(s, SessionStatus.COMPLETE, tmp_path)
        loaded = load_session(s.session_id, tmp_path)
        assert loaded is not None
        assert loaded.status == SessionStatus.COMPLETE


# ---------------------------------------------------------------------------
# summarize_session
# ---------------------------------------------------------------------------


class TestSummarizeSession:
    def _full_session(self, tmp_path: Path) -> ResearchSession:
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "canonical_ml_showcase",
                         data={"provider": "anthropic"}, sessions_base=tmp_path)
        s = record_event(s, SessionEventType.ITERATION_PROPOSAL_GENERATED,
                         "canonical_ml_showcase",
                         data={"context_hash": "ch1", "research_focus": "alpha sweep"},
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "canonical_ml_showcase",
                         data={"draft_id": "d1", "draft_hash": "dh1",
                               "proposed_name": "canonical_ml_showcase_v2"},
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.YAML_RENDERED, "canonical_ml_showcase",
                         data={"draft_id": "d1",
                               "config_path": "configs/experiments/canonical_ml_showcase_v2.yaml"},
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.EXPERIMENT_LINKED, "canonical_ml_showcase",
                         data={"new_experiment": "canonical_ml_showcase_v2"},
                         sessions_base=tmp_path)
        return s

    def test_returns_dict(self, tmp_path):
        s = _make_session(tmp_path)
        result = summarize_session(s)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self, tmp_path):
        s = _make_session(tmp_path)
        result = summarize_session(s)
        required = {
            "session_id", "research_goal", "status", "root_experiment",
            "active_experiment", "created_at", "updated_at", "event_count",
            "latest_review", "latest_proposal", "active_draft",
            "approved_config_path", "experiments_visited",
        }
        assert required <= set(result.keys())

    def test_event_count_correct(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        assert summarize_session(s)["event_count"] == 2

    def test_latest_review_extracted(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         data={"provider": "anthropic"}, sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["latest_review"] is not None
        assert result["latest_review"]["provider"] == "anthropic"

    def test_latest_review_is_most_recent(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         data={"provider": "stub"}, sessions_base=tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         data={"provider": "anthropic"}, sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["latest_review"]["provider"] == "anthropic"

    def test_latest_proposal_extracted(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.ITERATION_PROPOSAL_GENERATED, "exp_a",
                         data={"context_hash": "ch1", "research_focus": "rf1"},
                         sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["latest_proposal"] is not None
        assert result["latest_proposal"]["research_focus"] == "rf1"

    def test_active_draft_set_after_draft_generated(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "exp_a",
                         data={"draft_id": "d1", "draft_hash": "h",
                               "proposed_name": "exp_a_v2"},
                         sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["active_draft"] is not None
        assert result["active_draft"]["draft_id"] == "d1"

    def test_active_draft_cleared_after_yaml_rendered(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "exp_a",
                         data={"draft_id": "d1", "draft_hash": "h", "proposed_name": "v2"},
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.YAML_RENDERED, "exp_a",
                         data={"draft_id": "d1", "config_path": "configs/experiments/v2.yaml"},
                         sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["active_draft"] is None

    def test_approved_config_path_extracted(self, tmp_path):
        s = self._full_session(tmp_path)
        result = summarize_session(s)
        assert result["approved_config_path"] == \
            "configs/experiments/canonical_ml_showcase_v2.yaml"

    def test_approved_config_path_none_when_no_render(self, tmp_path):
        s = _make_session(tmp_path)
        result = summarize_session(s)
        assert result["approved_config_path"] is None

    def test_experiments_visited_ordered(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_b",
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["experiments_visited"] == ["exp_a", "exp_b"]

    def test_experiments_visited_unique(self, tmp_path):
        s = _make_session(tmp_path)
        for _ in range(5):
            s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                             sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["experiments_visited"] == ["exp_a"]

    def test_latest_review_none_when_no_reviews(self, tmp_path):
        s = _make_session(tmp_path)
        result = summarize_session(s)
        assert result["latest_review"] is None

    def test_latest_proposal_none_when_no_proposals(self, tmp_path):
        s = _make_session(tmp_path)
        result = summarize_session(s)
        assert result["latest_proposal"] is None

    def test_summary_is_pure_no_disk_io(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.REVIEW_GENERATED, "exp_a",
                         sessions_base=tmp_path)
        # Delete session file — summarize_session must not read from disk
        session_json_path(s.session_id, tmp_path).unlink()
        result = summarize_session(s)
        assert result["event_count"] == 1  # still works from in-memory session

    def test_active_draft_not_approved_after_generation(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "exp_a",
                         data={"draft_id": "d1", "draft_hash": "h", "proposed_name": "v2"},
                         sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["active_draft"]["approved"] is False

    def test_active_draft_shows_approved_after_approval(self, tmp_path):
        s = _make_session(tmp_path)
        s = record_event(s, SessionEventType.DRAFT_GENERATED, "exp_a",
                         data={"draft_id": "d1", "draft_hash": "h", "proposed_name": "v2"},
                         sessions_base=tmp_path)
        s = record_event(s, SessionEventType.DRAFT_APPROVED, "exp_a",
                         data={"draft_id": "d1", "draft_hash": "h"},
                         sessions_base=tmp_path)
        result = summarize_session(s)
        assert result["active_draft"]["approved"] is True


# ---------------------------------------------------------------------------
# list_session_ids
# ---------------------------------------------------------------------------


class TestListSessionIds:
    def test_empty_when_no_sessions(self, tmp_path):
        assert list_session_ids(tmp_path) == []

    def test_returns_created_session_ids(self, tmp_path):
        s1 = _make_session(tmp_path)
        s2 = create_session("exp_b", "goal b", sessions_base=tmp_path)
        ids = list_session_ids(tmp_path)
        assert s1.session_id in ids
        assert s2.session_id in ids

    def test_result_is_sorted(self, tmp_path):
        ids = []
        for i in range(3):
            s = create_session(f"exp_{i}", f"goal {i}", sessions_base=tmp_path)
            ids.append(s.session_id)
        listed = list_session_ids(tmp_path)
        assert listed == sorted(listed)


# ---------------------------------------------------------------------------
# Research API wrappers
# ---------------------------------------------------------------------------


class TestResearchApiWrappers:
    def test_create_research_session_exported(self):
        from src.orchestration.api.research_api import create_research_session
        assert callable(create_research_session)

    def test_load_research_session_exported(self):
        from src.orchestration.api.research_api import load_research_session
        assert callable(load_research_session)

    def test_record_session_event_exported(self):
        from src.orchestration.api.research_api import record_session_event
        assert callable(record_session_event)

    def test_update_research_session_status_exported(self):
        from src.orchestration.api.research_api import update_research_session_status
        assert callable(update_research_session_status)

    def test_summarize_research_session_exported(self):
        from src.orchestration.api.research_api import summarize_research_session
        assert callable(summarize_research_session)

    def test_list_research_sessions_exported(self):
        from src.orchestration.api.research_api import list_research_sessions
        assert callable(list_research_sessions)

    def test_create_via_api(self, tmp_path):
        from src.orchestration.api.research_api import create_research_session
        s = create_research_session("exp_a", "api goal", sessions_base=tmp_path)
        assert isinstance(s, ResearchSession)
        assert s.status == SessionStatus.ACTIVE

    def test_load_via_api(self, tmp_path):
        from src.orchestration.api.research_api import (
            create_research_session,
            load_research_session,
        )
        s = create_research_session("exp_a", "goal", sessions_base=tmp_path)
        loaded = load_research_session(s.session_id, tmp_path)
        assert loaded is not None
        assert loaded.session_id == s.session_id

    def test_record_event_via_api(self, tmp_path):
        from src.orchestration.api.research_api import (
            create_research_session,
            record_session_event,
        )
        s = create_research_session("exp_a", "goal", sessions_base=tmp_path)
        s = record_session_event(
            s, SessionEventType.REVIEW_GENERATED, "exp_a",
            data={"provider": "stub"}, sessions_base=tmp_path,
        )
        assert len(s.events) == 1

    def test_update_status_via_api(self, tmp_path):
        from src.orchestration.api.research_api import (
            create_research_session,
            update_research_session_status,
        )
        s = create_research_session("exp_a", "goal", sessions_base=tmp_path)
        s = update_research_session_status(s, SessionStatus.COMPLETE, tmp_path)
        assert s.status == SessionStatus.COMPLETE

    def test_summarize_via_api(self, tmp_path):
        from src.orchestration.api.research_api import (
            create_research_session,
            summarize_research_session,
        )
        s = create_research_session("exp_a", "goal", sessions_base=tmp_path)
        summary = summarize_research_session(s)
        assert summary["session_id"] == s.session_id

    def test_list_sessions_via_api(self, tmp_path):
        from src.orchestration.api.research_api import (
            create_research_session,
            list_research_sessions,
        )
        s = create_research_session("exp_a", "goal", sessions_base=tmp_path)
        ids = list_research_sessions(tmp_path)
        assert s.session_id in ids

    def test_orchestration_init_exports_session_functions(self):
        import src.orchestration as orch
        for fn in [
            "create_research_session", "load_research_session",
            "record_session_event", "update_research_session_status",
            "summarize_research_session", "list_research_sessions",
        ]:
            assert hasattr(orch, fn), f"Missing export: {fn}"


# ---------------------------------------------------------------------------
# Session __init__ exports
# ---------------------------------------------------------------------------


class TestSessionInitExports:
    def test_all_symbols_exported(self):
        from src.orchestration.session import (
            ResearchSession,
            SessionEvent,
            SessionEventType,
            SessionStatus,
            create_session,
            load_session,
            record_event,
            summarize_session,
            update_session_status,
        )
        assert all([
            ResearchSession, SessionEvent, SessionEventType, SessionStatus,
            create_session, load_session, record_event, summarize_session,
            update_session_status,
        ])


# ---------------------------------------------------------------------------
# Non-coupling: no quant engine, no evolution chain, no external artefacts
# ---------------------------------------------------------------------------


class TestNonCoupling:
    def test_create_session_imports_no_quant_engine(self):
        import src.orchestration.session.session_manager as sm
        import_lines = [
            line for line in open(sm.__file__).readlines() if line.startswith("import ") or line.startswith("from ")
        ]
        source = "".join(import_lines)
        for forbidden in ["src.experiments", "src.backtesting", "src.portfolio"]:
            assert forbidden not in source, f"Found forbidden import: {forbidden}"

    def test_create_session_imports_no_evolution_chain(self):
        import src.orchestration.session.session_manager as sm
        import_lines = [
            line for line in open(sm.__file__).readlines() if line.startswith("import ") or line.startswith("from ")
        ]
        source = "".join(import_lines)
        assert "evolution" not in source.lower()

    def test_summarize_session_pure_computation(self, tmp_path):
        """summarize_session should work even when no session files exist."""
        s = ResearchSession(
            session_id="detached",
            research_goal="no disk",
            root_experiment="exp_a",
            active_experiment="exp_a",
            status=SessionStatus.ACTIVE,
            created_at="2026-05-29T00:00:00+00:00",
            updated_at="2026-05-29T00:00:00+00:00",
            events=[
                SessionEvent(
                    event_id="e1",
                    event_type=SessionEventType.REVIEW_GENERATED,
                    timestamp="2026-05-29T00:00:00+00:00",
                    experiment_name="exp_a",
                    data={"provider": "stub"},
                )
            ],
        )
        result = summarize_session(s)
        assert result["event_count"] == 1
        assert result["latest_review"]["provider"] == "stub"

    def test_existing_api_functions_not_modified(self):
        """Spot-check that Phase 1-3 API functions still exist and are callable."""
        from src.orchestration.api.research_api import (
            approve_experiment_draft,
            build_research_evolution_chain,
            generate_experiment_draft,
            generate_iteration_proposal,
            render_draft_to_yaml,
            run_llm_review,
            validate_experiment_draft,
        )
        for fn in [
            run_llm_review, generate_iteration_proposal, generate_experiment_draft,
            validate_experiment_draft, approve_experiment_draft, render_draft_to_yaml,
            build_research_evolution_chain,
        ]:
            assert callable(fn)
