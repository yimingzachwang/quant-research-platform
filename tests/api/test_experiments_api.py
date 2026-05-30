"""Experiment discovery endpoint tests.

All backend functions are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.api.conftest import make_stub_summary

_MODULE = "src.orchestration.api.research_api"


class TestListExperiments:
    def test_returns_200(self, client):
        with patch(f"{_MODULE}.list_all_experiments", return_value=["exp_a", "exp_b"]):
            response = client.get("/api/experiments")
        assert response.status_code == 200

    def test_returns_experiment_list(self, client):
        with patch(f"{_MODULE}.list_all_experiments", return_value=["exp_a", "exp_b"]):
            data = client.get("/api/experiments").json()
        assert data["experiments"] == ["exp_a", "exp_b"]

    def test_empty_list(self, client):
        with patch(f"{_MODULE}.list_all_experiments", return_value=[]):
            data = client.get("/api/experiments").json()
        assert data["experiments"] == []


class TestGetSummary:
    def test_returns_200_when_found(self, client):
        stub = make_stub_summary()
        with patch(f"{_MODULE}.get_experiment_summary", return_value=stub):
            response = client.get("/api/experiments/exp_a/summary")
        assert response.status_code == 200

    def test_response_contains_summary(self, client):
        stub = make_stub_summary()
        with patch(f"{_MODULE}.get_experiment_summary", return_value=stub):
            data = client.get("/api/experiments/exp_a/summary").json()
        assert "summary" in data
        assert data["summary"]["experiment_name"] == "exp_a"
        assert data["summary"]["sharpe_ratio"] == 1.25

    def test_missing_experiment_returns_404(self, client):
        with patch(f"{_MODULE}.get_experiment_summary", return_value=None):
            response = client.get("/api/experiments/nonexistent/summary")
        assert response.status_code == 404

    def test_404_detail_mentions_experiment(self, client):
        with patch(f"{_MODULE}.get_experiment_summary", return_value=None):
            data = client.get("/api/experiments/nonexistent/summary").json()
        assert "nonexistent" in data["detail"]


class TestRankedExperiments:
    def test_returns_200(self, client):
        stub = make_stub_summary()
        with patch(f"{_MODULE}.rank_experiments_by_sharpe", return_value=[stub]):
            response = client.get("/api/experiments/ranked")
        assert response.status_code == 200

    def test_returns_experiment_list(self, client):
        stub = make_stub_summary()
        with patch(f"{_MODULE}.rank_experiments_by_sharpe", return_value=[stub]):
            data = client.get("/api/experiments/ranked").json()
        assert "experiments" in data
        assert len(data["experiments"]) == 1
        assert data["experiments"][0]["experiment_name"] == "exp_a"

    def test_default_by_sharpe(self, client):
        with patch(f"{_MODULE}.rank_experiments_by_sharpe", return_value=[]) as mock_rank:
            client.get("/api/experiments/ranked")
        mock_rank.assert_called_once()

    def test_limit_param_applied(self, client):
        stubs = [make_stub_summary(experiment_name=f"exp_{i}") for i in range(5)]
        with patch(f"{_MODULE}.rank_experiments_by_sharpe", return_value=stubs):
            data = client.get("/api/experiments/ranked?limit=3").json()
        assert len(data["experiments"]) == 3

    def test_unsupported_by_param_returns_400(self, client):
        response = client.get("/api/experiments/ranked?by=alpha")
        assert response.status_code == 400

    def test_400_detail_mentions_field(self, client):
        data = client.get("/api/experiments/ranked?by=alpha").json()
        assert "alpha" in data["detail"]

    def test_ranked_path_not_shadowed_by_name_param(self, client):
        # /api/experiments/ranked must NOT be matched as /api/experiments/{name}/summary
        with patch(f"{_MODULE}.rank_experiments_by_sharpe", return_value=[]):
            response = client.get("/api/experiments/ranked")
        assert response.status_code == 200


class TestNoCouplingExperiments:
    def test_no_quant_engine_import_in_experiments_router(self):
        import src.api.routers.experiments as mod
        import_lines = [
            line
            for line in open(mod.__file__).readlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        source = "".join(import_lines)
        for forbidden in ["src.experiments", "src.backtesting", "src.portfolio"]:
            assert forbidden not in source, f"Forbidden import found: {forbidden}"
