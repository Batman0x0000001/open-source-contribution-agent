"""验证Bash 工具的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio
from pathlib import Path

from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.tools.bash import BashTool, is_read_only_command
from osc_agent.processes.contracts import CommandResult


def context(root: Path) -> tool_context:
    return tool_context(
        session_id="bash", working_directory=str(root),
        state_directory=str(root / "state"),
    )


class Runner:
    request = None

    async def run(self, request):
        self.request = request
        return CommandResult(command=request.command, exit_code=0, stdout="ok\n", duration_ms=1)


def test_bash_tool_uses_non_shell_process_runner_contract(tmp_path: Path) -> None:
    runner = Runner()
    tool = BashTool(executable="/bin/bash", process_runner=runner)
    result = asyncio.run(tool.call(tool.input_model(command="python -m pytest"), context(tmp_path)))
    assert result.error is None and result.data["success"] is True
    assert runner.request.executable == "/bin/bash"
    assert runner.request.command == "python -m pytest"
    assert runner.request.environment.get("ANTHROPIC_API_KEY") is None


def test_bash_read_only_classification_is_narrow() -> None:
    assert is_read_only_command("rg needle .")
    assert not is_read_only_command("git status --short")
    assert not is_read_only_command("git diff --output=result.patch")
    assert not is_read_only_command("git diff --ext-diff")
    assert not is_read_only_command("git log -p --output=history.txt")
    assert not is_read_only_command("git show --ext-diff HEAD")
    assert not is_read_only_command("git push origin main")
    assert not is_read_only_command("rg needle . | tee result.txt")


def test_bash_hard_deny_precedes_execution(tmp_path: Path) -> None:
    runner = Runner()
    executor = ToolExecutor(ToolRegistry([
        BashTool(executable="/bin/bash", process_runner=runner)
    ]))
    result = asyncio.run(executor.execute(
        ToolUseBlock(id="danger", name="bash", input={"command": "sudo rm -rf /"}),
        context(tmp_path),
    ))
    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert runner.request is None


def test_bash_timeout_is_typed(tmp_path: Path) -> None:
    class TimeoutRunner:
        async def run(self, request):
            return CommandResult(
                command=request.command, exit_code=-1, stderr="timeout", duration_ms=1,
                termination_reason="timeout",
            )

    tool = BashTool(executable="/bin/bash", process_runner=TimeoutRunner())
    result = asyncio.run(tool.call(tool.input_model(command="python -m pytest"), context(tmp_path)))
    assert result.error and result.error.code == "BASH_TIMEOUT"
