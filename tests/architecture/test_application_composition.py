"""验证应用依赖组装的契约、边界条件与回归行为。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import AsyncIterator

import pytest
from typer.testing import CliRunner

import osc_agent.config as config_module
from osc_agent.composition import build_application
from osc_agent.cli import app
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.models import CapabilityScope, RuntimeMessage, TextBlock, ToolUseContext


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

    settings = config_module.load_settings()

    assert settings.model_id is None
    assert calls == [False]
    with pytest.raises(ValueError, match="MODEL_ID"):
        build_application(
            settings=settings,
            repo_root=tmp_path,
            model_gateway=CompletingGateway(),
        )


def test_subprocess_environment_allowlist_uses_strict_json_array(monkeypatch) -> None:
    monkeypatch.setenv(
        "OSC_AGENT_SUBPROCESS_ENV_ALLOWLIST",
        '["CUSTOM_BUILD_FLAG", "NODE_OPTIONS"]',
    )

    settings = config_module.Settings()

    assert settings.subprocess_env_allowlist == {
        "CUSTOM_BUILD_FLAG",
        "NODE_OPTIONS",
    }


def test_application_has_one_shared_runtime_and_executor_graph(tmp_path: Path) -> None:
    services = build_application(
        settings=Settings(model_id="test-model"),
        repo_root=tmp_path,
        model_gateway=CompletingGateway(),
    )

    assert services.runtime.dependencies.tool_executor is services.tool_executor
    assert services.runtime.dependencies.tool_registry is services.tool_registry
    assert services.agent_runner.runtime is services.runtime
    assert services.skill_executor.agent_runner is services.agent_runner
    assert services.skill_command_runner.executor is services.skill_executor
    assert services.skill_command_runner.runtime is services.runtime
    assert services.tool_registry.get("skill").executor is services.skill_executor
    assert services.tool_registry.get("agent").runner is services.agent_runner
    assert services.tool_registry.get("agent").registry is services.agent_registry
    assert [item.definition.name for item in services.agent_registry.list()] == ["explore", "verify"]
    assert services.agent_registry.get("verify").definition.config.max_rounds == 16
    assert services.agent_registry.get("explore").definition.config.max_rounds == 8
    assert services.query_config.max_rounds == 30
    assert "agent" in services.general_capabilities.allowed_tools
    assert "<available_skills>" in services.discovery_prompt
    assert "open-source-contribution" in services.discovery_prompt
    assert "<available_agents>" in services.discovery_prompt
    assert "- explore:" in services.discovery_prompt
    assert "- verify:" in services.discovery_prompt
    assert "<external_content_policy>" in services.discovery_prompt
    assert "not user authorization" in services.discovery_prompt
    assert [item.manifest.name for item in services.skill_catalog.list()] == [
        "issue-planning",
        "open-source-contribution",
    ]

    contribution = services.skill_catalog.get("open-source-contribution")
    contribution_capabilities = CapabilityScope(
        allowed_tools=contribution.manifest.allowed_tools
    )
    skill_discovery = services.skill_command_runner.discovery_prompt(
        contribution_capabilities
    )
    assert "- explore:" in skill_discovery
    assert "- verify:" in skill_discovery

    general_context = ToolUseContext(
        session_id="general",
        working_directory=str(tmp_path),
        repository_root=str(tmp_path),
        state_directory=str(tmp_path / ".state"),
        capabilities=services.general_capabilities,
    )
    contribution_context = general_context.model_copy(
        update={"capabilities": contribution_capabilities}
    )
    assert "agent" in {
        schema["name"] for schema in services.tool_registry.schemas(general_context)
    }
    assert "agent" in {
        schema["name"] for schema in services.tool_registry.schemas(contribution_context)
    }


def test_cli_exposes_only_new_architecture_commands(tmp_path: Path) -> None:
    runner = CliRunner()

    root_help = runner.invoke(app, ["--help"])
    skill_help = runner.invoke(app, ["skill", "--help"])
    contribution_help = runner.invoke(app, ["contribute", "--help"])
    listed = runner.invoke(app, ["skill", "list", "--repo", str(tmp_path)])

    assert root_help.exit_code == 0
    assert skill_help.exit_code == 0
    assert contribution_help.exit_code == 0
    assert listed.exit_code == 0
    assert "run" in root_help.stdout
    assert "--repo-url" in contribution_help.stdout
    assert "resume" in root_help.stdout
    assert "open-source-contribution" in listed.stdout


def test_cli_does_not_import_legacy_runtime_or_stage_functions() -> None:
    path = PROJECT_ROOT / "osc_agent" / "cli.py"
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


def test_product_entrypoints_use_agent_application_service_only() -> None:
    for relative in ("osc_agent/cli.py", "osc_agent/bot/worker.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "build_agent_application" in called_names
        assert "build_application" not in called_names
        assert "StartQueryParams" not in source
        assert "ResumeQueryParams" not in source
        assert ".runtime.query(" not in source
