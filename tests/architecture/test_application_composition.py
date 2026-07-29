"""验证应用依赖组装的契约、边界条件与回归行为。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import AsyncIterator

import pytest
from typer.testing import CliRunner

import osc_agent.config as config_module
from osc_agent.application import AgentApplicationConfig, AgentProfile
from osc_agent.application.composition import build_discovery_prompt, compose_application
from osc_agent.cli import app
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.messages import RuntimeMessage, TextBlock
from osc_agent.runtime.tool_models import CapabilityScope, ToolUseContext


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
        compose_application(
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

    settings = config_module.Settings()

    assert settings.subprocess_env_allowlist == {
        "CUSTOM_BUILD_FLAG",
        "NODE_OPTIONS",
    }


def test_application_has_one_shared_runtime_and_executor_graph(tmp_path: Path) -> None:
    graph = compose_application(
        AgentApplicationConfig(
            settings=Settings(model_id="test-model"),
            repository_root=tmp_path,
            profile=AgentProfile(profile_id="test", system_prompt="test"),
            model_gateway=CompletingGateway(),
        )
    )

    assert graph.runtime.dependencies.tool_executor is graph.tool_executor
    assert graph.runtime.dependencies.tool_executor.registry is graph.tool_registry
    assert graph.subagent_runner.runtime is graph.runtime
    assert graph.tool_registry.get("skill").preparer is graph.skill_preparer
    assert graph.tool_registry.get("agent").runner is graph.subagent_runner
    assert graph.tool_registry.get("agent").registry is graph.subagent_registry
    assert [item.definition.name for item in graph.subagent_registry.list()] == ["explore", "verify"]
    assert graph.subagent_registry.get("verify").definition.config.max_rounds == 16
    assert graph.subagent_registry.get("explore").definition.config.max_rounds == 8
    assert graph.query_config.max_rounds == 30
    assert "agent" in graph.general_capabilities.allowed_tools
    assert "<available_skills>" in graph.discovery_prompt
    assert "open-source-contribution" in graph.discovery_prompt
    assert "When to use:" in graph.discovery_prompt
    assert "<available_agents>" in graph.discovery_prompt
    assert "- explore:" in graph.discovery_prompt
    assert "- verify:" in graph.discovery_prompt
    assert "<external_content_policy>" in graph.discovery_prompt
    assert "not user authorization" in graph.discovery_prompt
    assert [item.manifest.name for item in graph.skill_catalog.list()] == [
        "issue-planning",
        "open-source-contribution",
    ]

    contribution = graph.skill_catalog.get("open-source-contribution")
    contribution_capabilities = CapabilityScope(
        allowed_tools=contribution.manifest.allowed_tools
    )
    skill_discovery = build_discovery_prompt(
        graph.skill_catalog, graph.subagent_registry, contribution_capabilities
    )
    assert "- explore:" in skill_discovery
    assert "- verify:" in skill_discovery

    general_context = ToolUseContext(
        session_id="general",
        working_directory=str(tmp_path),
        state_directory=str(tmp_path / ".state"),
        capabilities=graph.general_capabilities,
    )
    contribution_context = general_context.model_copy(
        update={"capabilities": contribution_capabilities}
    )
    assert "agent" in {
        schema["name"] for schema in graph.tool_registry.schemas(general_context)
    }
    assert "agent" in {
        schema["name"] for schema in graph.tool_registry.schemas(contribution_context)
    }


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
    assert "--repo-url" in contribution_help.stdout
    assert "resume" in root_help.stdout
    assert "open-source-contribution" in listed.stdout
    assert "INVALID_SKILL_MANIFEST" in listed.stderr


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


def test_product_entrypoints_use_agent_application_only() -> None:
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
        assert "AgentRunSpec" not in source
        assert "RunEnvironment" not in source
        assert "StartQueryParams" not in source
        assert "ResumeQueryParams" not in source
        assert ".runtime.query(" not in source


def test_application_composition_does_not_name_bot_artifact_tools() -> None:
    source = (PROJECT_ROOT / "osc_agent" / "application" / "composition.py").read_text(
        encoding="utf-8"
    )

    assert "submit_issue_plan" not in source
    assert "submit_delivery_draft" not in source
