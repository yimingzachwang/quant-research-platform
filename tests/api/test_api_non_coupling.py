"""Non-coupling verification tests for the API bridge.

Verify that:
  - No router imports quant engine internals directly
  - No endpoint executes experiments
  - No endpoint registers lineage automatically
  - No endpoint creates autonomous loops
  - The app imports cleanly without side effects
"""

from __future__ import annotations

import importlib

import pytest

_ROUTER_MODULES = [
    "src.api.routers.sessions",
    "src.api.routers.reviews",
    "src.api.routers.drafts",
    "src.api.routers.routing",
    "src.api.routers.experiments",
]

_FORBIDDEN_QUANT_IMPORTS = [
    "src.experiments",
    "src.backtesting",
    "src.portfolio",
    "src.data",
]

_FORBIDDEN_ORCHESTRATION_UTILS = [
    "src.orchestration.utils.filesystem",
    "src.orchestration.utils.serialization",
]

_FORBIDDEN_EXECUTION_PATTERNS = [
    "run_experiment",
    "execute_experiment",
    "orchestrator.run",
    "register_lineage",
    "register_experiment_lineage",
]


def _import_lines(module_name: str) -> str:
    mod = importlib.import_module(module_name)
    lines = open(mod.__file__).readlines()
    return "".join(
        line for line in lines
        if line.startswith("import ") or line.startswith("from ")
    )


def _all_lines(module_name: str) -> str:
    mod = importlib.import_module(module_name)
    return open(mod.__file__).read()


class TestNoQuantEngineImports:
    @pytest.mark.parametrize("module_name", _ROUTER_MODULES)
    def test_no_quant_engine_import(self, module_name):
        import_source = _import_lines(module_name)
        for forbidden in _FORBIDDEN_QUANT_IMPORTS:
            assert forbidden not in import_source, (
                f"{module_name} imports '{forbidden}' — quant engine must not be called from router"
            )

    def test_app_does_not_import_quant_engine(self):
        import_source = _import_lines("src.api.app")
        for forbidden in _FORBIDDEN_QUANT_IMPORTS:
            assert forbidden not in import_source


class TestNoOrchestrationUtilImports:
    @pytest.mark.parametrize("module_name", _ROUTER_MODULES)
    def test_no_direct_orchestration_util_import(self, module_name):
        import_source = _import_lines(module_name)
        for forbidden in _FORBIDDEN_ORCHESTRATION_UTILS:
            assert forbidden not in import_source, (
                f"{module_name} imports '{forbidden}' directly — "
                "routers must call only Research API functions, not internal utilities"
            )


class TestNoExperimentExecution:
    @pytest.mark.parametrize("module_name", _ROUTER_MODULES)
    def test_no_execution_call_in_router(self, module_name):
        source = _all_lines(module_name)
        for pattern in _FORBIDDEN_EXECUTION_PATTERNS:
            assert pattern not in source, (
                f"{module_name} contains '{pattern}' — experiment execution must not be invoked from router"
            )


class TestNoLineageRegistration:
    @pytest.mark.parametrize("module_name", _ROUTER_MODULES)
    def test_no_automatic_lineage_registration(self, module_name):
        source = _all_lines(module_name)
        assert "register_lineage" not in source, (
            f"{module_name} calls register_lineage — lineage is human-triggered only"
        )


class TestNoAutonomousWorkflow:
    @pytest.mark.parametrize("module_name", _ROUTER_MODULES)
    def test_no_background_tasks_in_router(self, module_name):
        source = _all_lines(module_name)
        for pattern in ["BackgroundTasks", "asyncio.create_task", "threading.Thread", "subprocess"]:
            assert pattern not in source, (
                f"{module_name} uses '{pattern}' — no autonomous background execution allowed"
            )

    def test_no_autonomous_loop_in_app(self):
        source = _all_lines("src.api.app")
        for pattern in ["BackgroundTasks", "asyncio.create_task", "threading.Thread"]:
            assert pattern not in source


class TestAppStructure:
    def test_app_imports_successfully(self):
        from src.api.app import app
        assert app is not None

    def test_health_endpoint_exists(self):
        from src.api.app import app
        paths = [r.path for r in app.routes]
        assert "/api/health" in paths

    def test_no_database_imports(self):
        for module_name in _ROUTER_MODULES + ["src.api.app"]:
            source = _all_lines(module_name)
            for pattern in ["sqlalchemy", "pymongo", "psycopg", "sqlite3", "redis"]:
                assert pattern not in source, (
                    f"{module_name} imports database layer '{pattern}'"
                )

    def test_no_auth_imports(self):
        for module_name in _ROUTER_MODULES + ["src.api.app"]:
            source = _all_lines(module_name)
            for pattern in ["jwt", "oauth2", "passlib", "bcrypt", "HTTPBasic"]:
                assert pattern.lower() not in source.lower(), (
                    f"{module_name} imports auth dependency '{pattern}'"
                )
