from __future__ import annotations

import ast
from pathlib import Path
from typing import AsyncIterator

from typer.testing import CliRunner

from osc_agent.application import build_application
from osc_agent.cli import app
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.models import RuntimeMessage, TextBlock


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CompletingGateway:
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        yield ModelCompleted(
            message=RuntimeMessage(role="assistant", content=[TextBlock(text="done")]),
            stop_reason="end_turn",
        )


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
    assert "<available_skills>" in services.discovery_prompt
    assert "open-source-contribution" in services.discovery_prompt
    assert "<available_agents>" in services.discovery_prompt
    assert "general" in services.discovery_prompt
    assert [item.manifest.name for item in services.skill_catalog.list()] == ["open-source-contribution"]


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
