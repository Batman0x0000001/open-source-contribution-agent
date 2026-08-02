"""验证应用依赖组装的契约、边界条件与回归行为。"""

from __future__ import annotations

from osc_agent.runtime.state import CapabilityScope

from tests.runtime_factories import tool_context

import ast
from pathlib import Path
from typing import AsyncIterator

import pytest
from typer.testing import CliRunner

import osc_agent.configuration.agent as config_module
from osc_agent.application import AgentApplicationConfig, AgentProfile, build_agent_application
from osc_agent.cli.app import app
from tests.settings_factory import make_agent_settings as Settings
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.messages import RuntimeMessage, TextBlock



PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CompletingGateway:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelCompleted(
            message=RuntimeMessage(role="assistant", content=[TextBlock(text="done")]),
            stop_reason="end_turn",
        )


def test_model_id_is_explicit_and_dotenv_does_not_override_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        config_module,
        "load_dotenv",
        lambda *, override: calls.append(override),
    )
    monkeypatch.delenv("MODEL_ID", raising=False)

    settings = config_module.load_agent_settings()

    assert settings.model_id is None
    assert calls == [False]
    with pytest.raises(ValueError, match="MODEL_ID"):
        build_agent_application(
            AgentApplicationConfig(
                settings=settings,
                repository_root=tmp_path,
                profile=AgentProfile(profile_id="test", system_prompt="test"),
                model_gateway=CompletingGateway(),
            )
        )


def test_subprocess_environment_allowlist_uses_strict_json_array(monkeypatch) -> None:
    monkeypatch.setenv(
        "OSC_AGENT_SUBPROCESS_ENV_ALLOWLIST",
        '["CUSTOM_BUILD_FLAG", "NODE_OPTIONS"]',
    )

    settings = config_module.load_agent_settings()

    assert settings.subprocess_env_allowlist == {
        "CUSTOM_BUILD_FLAG",
        "NODE_OPTIONS",
    }


def test_application_has_one_shared_runtime_and_executor_graph(tmp_path: Path) -> None:
    application = build_agent_application(
        AgentApplicationConfig(
            settings=Settings(model_id="test-model"),
            repository_root=tmp_path,
            profile=AgentProfile(profile_id="test", system_prompt="test"),
            model_gateway=CompletingGateway(),
        )
    )

    executor = application._runtime.dependencies.tool_executor
    registry = executor.registry
    agent_tool = registry.get("agent")
    skill_tool = registry.get("skill")
    subagents = agent_tool.registry
    catalog = application._skill_preparer.catalog
    assert agent_tool.runner.runtime is application._runtime
    assert skill_tool.preparer is application._skill_preparer
    assert [item.definition.name for item in subagents.list()] == ["explore", "verify"]
    assert subagents.get("verify").definition.config.max_rounds == 16
    assert subagents.get("explore").definition.config.max_rounds == 8
    assert application._query_config.max_rounds == 30
    assert "agent" in application._capabilities.allowed_tools
    assert not hasattr(application, "_discovery_prompt")
    assert "open-source-contribution" in skill_tool.description
    assert "explore" in agent_tool.description
    assert "verify" in agent_tool.description
    assert [item.manifest.name for item in catalog.list()] == [
        "issue-planning",
        "open-source-contribution",
    ]

    contribution = catalog.get("open-source-contribution")
    contribution_capabilities = CapabilityScope(
        allowed_tools=contribution.manifest.allowed_tools
    )
    general_context = tool_context(
        session_id="general",
        working_directory=str(tmp_path),
        state_directory=str(tmp_path / ".state"),
        capabilities=application._capabilities,
    )
    contribution_context = general_context.model_copy(
        update={"capabilities": contribution_capabilities}
    )
    assert "agent" in {
        schema["name"] for schema in registry.schemas(general_context)
    }
    assert "agent" in {
        schema["name"] for schema in registry.schemas(contribution_context)
    }


def test_profile_capabilities_are_resolved_once_for_schema_and_execution(tmp_path: Path) -> None:
    application = build_agent_application(
        AgentApplicationConfig(
            settings=Settings(model_id="test-model"),
            repository_root=tmp_path,
            profile=AgentProfile(
                profile_id="read-only",
                system_prompt="Read only.",
                allowed_tools=frozenset({"read_file"}),
            ),
            model_gateway=CompletingGateway(),
        )
    )
    context = tool_context(
        session_id="restricted",
        working_directory=str(tmp_path),
        state_directory=str(tmp_path / ".state"),
        capabilities=application._capabilities,
    )

    assert application._capabilities.allowed_tools == frozenset({"read_file"})
    assert [
        schema["name"]
        for schema in application._runtime.dependencies.tool_executor.registry.schemas(context)
    ] == ["read_file"]


def test_cli_exposes_only_new_architecture_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    invalid = tmp_path / ".osc_agent" / "skills" / "legacy" / "SKILL.md"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(
        """---
name: legacy
description: Legacy
when_to_use: Never
input_schema: {type: object}
---
Legacy.
""",
        encoding="utf-8",
    )

    root_help = runner.invoke(app, ["--help"])
    skill_help = runner.invoke(app, ["skill", "--help"])
    contribution_help = runner.invoke(app, ["contribute", "--help"])
    listed = runner.invoke(app, ["skill", "list", "--repo", str(tmp_path)])

    assert root_help.exit_code == 0
    assert skill_help.exit_code == 0
    assert contribution_help.exit_code == 0
    assert listed.exit_code == 0
    assert "run" in root_help.stdout
    assert "bot" not in root_help.stdout
    assert "deploy" not in root_help.stdout
    assert "architecture" not in root_help.stdout
    assert "--repo-url" in contribution_help.stdout
    assert "resume" in root_help.stdout
    assert "open-source-contribution" in listed.stdout
    assert "INVALID_SKILL_MANIFEST" in listed.stderr


def test_cli_does_not_import_legacy_runtime_or_stage_functions() -> None:
    path = PROJECT_ROOT / "osc_agent" / "cli" / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "osc_agent.agent_loop" not in imports
    assert not any(module.startswith("osc_agent.harness") for module in imports)
    source = path.read_text(encoding="utf-8")
    for old_name in ("discover_stage", "design_stage", "implement_stage", "draft_pr_stage"):
        assert old_name not in source


def test_product_entrypoints_use_agent_application_only() -> None:
    factories = ("osc_agent/cli/agent.py", "osc_agent/bot/worker/agent_jobs.py")
    for relative in factories:
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "build_agent_application" in called_names
        assert "build_application" not in called_names

    entrypoints = (
        "osc_agent/cli/app.py",
        "osc_agent/bot/worker/coordinator.py",
    )
    for relative in entrypoints:
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "AgentRunSpec" not in source
        assert "RunEnvironment" not in source
        assert "StartQueryParams" not in source
        assert "ResumeQueryParams" not in source
        assert ".runtime.query(" not in source


def test_cli_and_bot_entrypoints_have_separate_product_boundaries() -> None:
    cli_root = PROJECT_ROOT / "osc_agent" / "cli"
    cli_imports = {
        node.module
        for path in cli_root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    bot_entrypoint = PROJECT_ROOT / "osc_agent" / "bot" / "entrypoint.py"
    bot_imports = {
        node.module
        for node in ast.walk(ast.parse(bot_entrypoint.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("osc_agent.bot") for module in cli_imports)
    assert not any(module.startswith("osc_agent.cli") for module in bot_imports)
    for legacy in ("cli.py", "config.py", "cli_session.py", "runtime_config.py"):
        assert not (PROJECT_ROOT / "osc_agent" / legacy).exists()


def test_control_service_does_not_load_worker_or_agent_execution() -> None:
    path = PROJECT_ROOT / "osc_agent" / "bot" / "control" / "service.py"
    imports = {
        node.module
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    forbidden = (
        "osc_agent.application",
        "osc_agent.providers",
        "osc_agent.bot.worker",
    )
    assert not any(module.startswith(forbidden) for module in imports)


def test_bot_process_packages_have_one_way_dependencies() -> None:
    bot_root = PROJECT_ROOT / "osc_agent" / "bot"
    control_violations = [
        f"{path.relative_to(PROJECT_ROOT)} -> {node.module}"
        for path in (bot_root / "control").rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(("osc_agent.bot.worker", "osc_agent.application", "osc_agent.providers"))
    ]
    worker_violations = [
        f"{path.relative_to(PROJECT_ROOT)} -> {node.module}"
        for path in (bot_root / "worker").rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("osc_agent.bot.control")
    ]

    assert not control_violations
    assert not worker_violations


def test_application_composition_does_not_name_bot_artifact_tools() -> None:
    source = (PROJECT_ROOT / "osc_agent" / "application" / "agent.py").read_text(
        encoding="utf-8"
    )

    assert "submit_issue_plan" not in source
    assert "submit_delivery_draft" not in source


def test_application_and_doctor_share_the_default_subagent_set() -> None:
    application = (
        PROJECT_ROOT / "osc_agent" / "application" / "agent.py"
    ).read_text(encoding="utf-8")
    doctor = (PROJECT_ROOT / "osc_agent" / "cli" / "doctor.py").read_text(
        encoding="utf-8"
    )

    assert "build_default_subagents" in application
    assert "build_default_subagents" in doctor
    assert "build_explore_subagent" not in doctor
    assert "build_verify_subagent" not in doctor


def test_session_summary_rendering_has_one_cli_owner() -> None:
    driver = (PROJECT_ROOT / "osc_agent" / "cli" / "agent.py").read_text(
        encoding="utf-8"
    )
    sessions = (PROJECT_ROOT / "osc_agent" / "cli" / "sessions.py").read_text(
        encoding="utf-8"
    )

    assert "render_session_summary(" in driver
    assert "def render_session_summary(" not in driver
    assert "def render_session_summary(" in sessions
