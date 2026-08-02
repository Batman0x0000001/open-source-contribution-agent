"""验证文本搜索工具的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio
from pathlib import Path

import pytest

import osc_agent.tools.search as search_module
from osc_agent.processes.contracts import CommandResult
from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.tools.search import GrepTool


def context(root: Path) -> tool_context:
    return tool_context(
        session_id="grep",
        working_directory=str(root),
        state_directory=str(root / "state"),
    )


def test_grep_returns_bounded_structured_matches_and_honors_ignore(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("Needle\nneedle\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Needle\n", encoding="utf-8")
    tool = GrepTool()
    executor = ToolExecutor(ToolRegistry([tool]))

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="grep",
                name="grep",
                input={
                    "pattern": "needle",
                    "case_sensitive": False,
                    "max_matches": 1,
                },
            ),
            context(tmp_path),
        )
    )

    assert result.error is None
    assert result.data["matches"] == [
        {"path": "visible.txt", "line": 1, "text": "Needle"}
    ]
    assert result.data["truncated"] is True


def test_grep_rejects_path_escape_and_option_injection(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([GrepTool()]))

    escaped = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="escape",
                name="grep",
                input={"pattern": "x", "path": "../outside"},
            ),
            context(tmp_path),
        )
    )
    injected = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="inject",
                name="grep",
                input={"pattern": "x", "glob": "--pre=evil"},
            ),
            context(tmp_path),
        )
    )

    assert escaped.error and escaped.error.code == "TOOL_VALIDATION_FAILED"
    assert injected.error and injected.error.code == "TOOL_VALIDATION_FAILED"


def test_grep_uses_the_cancellable_trusted_argument_runner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    requests = []

    async def run_arguments(executable, arguments, **kwargs):
        requests.append((executable, arguments, kwargs))
        return CommandResult(
            command="rg",
            exit_code=0,
            stdout=(
                '{"type":"match","data":{"path":{"text":"visible.txt"},'
                '"lines":{"text":"needle\\n"},"line_number":1}}\n'
            ),
            duration_ms=1,
        )

    monkeypatch.setattr(search_module, "run_arguments", run_arguments)
    tool = GrepTool(executable="trusted-rg")

    result = asyncio.run(
        tool.call(tool.input_model(pattern="needle"), context(tmp_path))
    )

    assert result.error is None
    assert result.data["matches"][0]["path"] == "visible.txt"
    assert requests[0][0] == "trusted-rg"
    assert requests[0][2]["repo_root"] == tmp_path


def test_grep_propagates_trusted_runner_cancellation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def run_arguments(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(search_module, "run_arguments", run_arguments)
    tool = GrepTool(executable="trusted-rg")

    async def exercise() -> None:
        task = asyncio.create_task(
            tool.call(tool.input_model(pattern="needle"), context(tmp_path))
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
