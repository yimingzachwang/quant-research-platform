"""Review and iteration proposal endpoint tests.

All backend functions are mocked.
"""

from __future__ import annotations

from unittest.mock import call, patch

from tests.api.conftest import (
    make_stub_comparison,
    make_stub_proposal,
    make_stub_review,
    make_stub_session,
)

_MODULE = "src.orchestration.api.research_api"


class TestReviewEndpoint:
    def _post_review(self, client, stub_session, stub_review, updated_session=None):
        updated = updated_session or stub_session
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub_session),
            patch(f"{_MODULE}.run_llm_review", return_value=stub_review),
            patch(f"{_MODULE}.record_session_event", return_value=updated),
        ):
            return client.post(
                "/api/sessions/test-session-id/review",
                json={"experiment_name": "exp_a", "provider": "stub"},
            )

    def test_returns_200(self, client):
        assert self._post_review(client, make_stub_session(), make_stub_review()).status_code == 200

    def test_response_contains_review(self, client):
        stub_review = make_stub_review()
        data = self._post_review(client, make_stub_session(), stub_review).json()
        assert "review" in data
        assert data["review"]["review_text"] == "Strong OOS stability observed."

    def test_response_contains_session_and_summary(self, client):
        data = self._post_review(client, make_stub_session(), make_stub_review()).json()
        assert "session" in data
        assert "summary" in data

    def test_records_review_generated_event(self, client):
        stub = make_stub_session()
        review = make_stub_review()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.run_llm_review", return_value=review) as mock_review,
            patch(f"{_MODULE}.record_session_event", return_value=stub) as mock_record,
        ):
            client.post(
                "/api/sessions/test-session-id/review",
                json={"experiment_name": "exp_a", "provider": "stub"},
            )
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args
        assert call_kwargs.kwargs["event_type"] == "REVIEW_GENERATED"

    def test_missing_session_returns_404(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            response = client.post(
                "/api/sessions/ghost/review",
                json={"experiment_name": "exp_a"},
            )
        assert response.status_code == 404


class TestProposalEndpoint:
    def _post_proposal(self, client, stub_session, stub_proposal):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub_session),
            patch(f"{_MODULE}.generate_iteration_proposal", return_value=stub_proposal),
            patch(f"{_MODULE}.record_session_event", return_value=stub_session),
        ):
            return client.post(
                "/api/sessions/test-session-id/proposal",
                json={"experiment_name": "exp_a", "provider": "stub"},
            )

    def test_returns_200(self, client):
        assert self._post_proposal(
            client, make_stub_session(), make_stub_proposal()
        ).status_code == 200

    def test_response_contains_proposal(self, client):
        data = self._post_proposal(
            client, make_stub_session(), make_stub_proposal()
        ).json()
        assert "proposal" in data
        assert data["proposal"]["research_focus"] == "Regularisation sweep"

    def test_response_contains_session_and_summary(self, client):
        data = self._post_proposal(
            client, make_stub_session(), make_stub_proposal()
        ).json()
        assert "session" in data
        assert "summary" in data

    def test_records_proposal_generated_event(self, client):
        stub = make_stub_session()
        proposal = make_stub_proposal()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.generate_iteration_proposal", return_value=proposal),
            patch(f"{_MODULE}.record_session_event", return_value=stub) as mock_record,
        ):
            client.post(
                "/api/sessions/test-session-id/proposal",
                json={"experiment_name": "exp_a", "provider": "stub"},
            )
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["event_type"] == "ITERATION_PROPOSAL_GENERATED"

    def test_missing_session_returns_404(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            response = client.post(
                "/api/sessions/ghost/proposal",
                json={"experiment_name": "exp_a"},
            )
        assert response.status_code == 404


class TestCompareEndpoint:
    def _post_compare(self, client, stub_session, stub_comparison):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub_session),
            patch(f"{_MODULE}.run_llm_comparative_review", return_value=stub_comparison),
        ):
            return client.post(
                "/api/sessions/test-session-id/compare",
                json={"baseline": "exp_a", "candidate": "exp_b", "provider": "stub"},
            )

    def test_returns_200(self, client):
        assert self._post_compare(
            client, make_stub_session(), make_stub_comparison()
        ).status_code == 200

    def test_response_contains_comparison(self, client):
        data = self._post_compare(
            client, make_stub_session(), make_stub_comparison()
        ).json()
        assert "comparison" in data
        assert data["comparison"]["baseline_experiment"] == "exp_a"
        assert data["comparison"]["candidate_experiment"] == "exp_b"

    def test_missing_session_returns_404(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            response = client.post(
                "/api/sessions/ghost/compare",
                json={"baseline": "exp_a", "candidate": "exp_b"},
            )
        assert response.status_code == 404

    def test_compare_does_not_record_session_event(self, client):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
            patch(
                f"{_MODULE}.run_llm_comparative_review",
                return_value=make_stub_comparison(),
            ),
            patch(f"{_MODULE}.record_session_event") as mock_record,
        ):
            client.post(
                "/api/sessions/test-session-id/compare",
                json={"baseline": "exp_a", "candidate": "exp_b"},
            )
        mock_record.assert_not_called()
