from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

from osc_agent.runtime.models import ToolUseBlock, ToolUseContext
from osc_agent.runtime.tool_execution import ToolExecutor
from tests.contracts.registry_factory import build_test_tool_registry as build_core_tool_registry
from osc_agent.tools.git import GitDiffInput, GitLogInput, GitStatusInput


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    (root / "tracked.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)


def context(root: Path) -> ToolUseContext:
    return ToolUseContext(session_id="session-1", working_directory=str(root), repository_root=str(root), state_directory=str(root / "state"))


def test_git_tools_have_strict_contracts_and_read_only_concurrency() -> None:
    registry = build_core_tool_registry()
    status = registry.get("git_status")
    diff = registry.get("git_diff")
    log = registry.get("git_log")

    assert status.input_model is GitStatusInput
    assert diff.input_model is GitDiffInput
    assert log.input_model is GitLogInput
    assert status.is_read_only(GitStatusInput()) is True
    assert diff.is_concurrency_safe(GitDiffInput()) is True
    assert log.is_concurrency_safe(GitLogInput()) is True


def test_git_tools_return_structured_status_diff_and_log(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    (tmp_path / "tracked.txt").write_text("after\n", encoding="utf-8")
    executor = ToolExecutor(build_core_tool_registry())

    async def execute(name: str, input: dict):
        return await executor.execute(
            ToolUseBlock(id=name, name=name, input=input),
            context(tmp_path),
        )

    status, diff, log = asyncio.run(
        _gather(
            execute("git_status", {}),
            execute("git_diff", {}),
            execute("git_log", {"limit": 1}),
        )
    )

    assert status.data == {"status": " M tracked.txt"}
    assert "-before" in diff.data["diff"]
    assert "+after" in diff.data["diff"]
    assert "initial" in log.data["log"]


async def _gather(*calls):
    return await asyncio.gather(*calls)


def test_git_log_rejects_implicit_and_out_of_range_limits(tmp_path: Path) -> None:
    executor = ToolExecutor(build_core_tool_registry())

    implicit = asyncio.run(
        executor.execute(
            ToolUseBlock(id="log-1", name="git_log", input={"limit": "5"}),
            context(tmp_path),
        )
    )
    too_large = asyncio.run(
        executor.execute(
            ToolUseBlock(id="log-2", name="git_log", input={"limit": 51}),
            context(tmp_path),
        )
    )

    assert implicit.error and implicit.error.code == "TOOL_INPUT_INVALID"
    assert too_large.error and too_large.error.code == "TOOL_INPUT_INVALID"


def test_git_command_failure_is_structured(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    (isolated / ".git").write_text("gitdir: missing-git-directory\n", encoding="utf-8")
    result = asyncio.run(
        ToolExecutor(build_core_tool_registry()).execute(
            ToolUseBlock(id="status-1", name="git_status", input={}),
            context(isolated),
        )
    )

    assert result.error and result.error.code == "GIT_COMMAND_FAILED"
