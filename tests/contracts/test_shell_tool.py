from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.models import Ask, ToolUseBlock, ToolUseContext
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor
from tests.contracts.registry_factory import build_test_tool_registry as build_core_tool_registry
from osc_agent.tools.shell import ShellInput, ShellTool


def context(root: Path) -> ToolUseContext:
    return ToolUseContext(session_id="session-1", working_directory=str(root), repository_root=str(root), state_directory=str(root / "state"))


def test_shell_classifies_behavior_from_each_input() -> None:
    tool = ShellTool()

    assert tool.name == "powershell"
    assert Path(tool.executable).stem.casefold() == "pwsh"
    assert tool.is_read_only(ShellInput(command="git status")) is True
    assert tool.is_concurrency_safe(ShellInput(command="rg pattern src")) is True
    assert tool.is_read_only(ShellInput(command="git status && git push")) is False
    assert tool.is_destructive(ShellInput(command="python build.py")) is True


def test_shell_read_only_classifier_rejects_execution_writes_and_path_escape() -> None:
    tool = ShellTool()

    assert tool.is_read_only(ShellInput(command="rg --pre malicious pattern")) is False
    assert tool.is_read_only(ShellInput(command="git diff --output=result.patch")) is False
    assert tool.is_read_only(ShellInput(command="Get-Content ..\\secret.txt")) is False
    assert tool.is_read_only(ShellInput(command="Get-Content C:\\secret.txt")) is False


def test_read_only_shell_executes_without_approval(tmp_path: Path) -> None:
    executor = ToolExecutor(build_core_tool_registry())

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="shell-1", name="powershell", input={"command": "git status --short"}),
            context(tmp_path),
        )
    )

    assert result.error is None
    assert "output" in result.data


def test_non_read_only_shell_requires_one_explicit_approval(tmp_path: Path) -> None:
    approvals = 0

    async def approve(decision: Ask) -> bool:
        nonlocal approvals
        approvals += 1
        return True

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="shell-1", name="powershell", input={"command": "python -c \"print('ok')\""}),
            context(tmp_path),
        )
    )

    assert approvals == 1
    assert result.error is None
    assert "ok" in result.data["output"]


def test_denied_shell_pattern_is_rejected_before_permission_prompt(tmp_path: Path) -> None:
    approvals = 0

    async def approve(decision: Ask) -> bool:
        nonlocal approvals
        approvals += 1
        return True

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="shell-1", name="powershell", input={"command": "git push origin main"}),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert approvals == 0


def test_powershell_process_does_not_use_shell_true() -> None:
    source = (Path(__file__).resolve().parents[2] / "osc_agent" / "tools" / "process.py").read_text(encoding="utf-8")
    assert "create_subprocess_exec" in source
    assert "shell=True" not in source
