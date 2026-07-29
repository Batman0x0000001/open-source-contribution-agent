"""验证Git 工具的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio
from pathlib import Path
import shutil
import subprocess

import pytest

from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.workspaces.models import WorktreeSession
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.session_store import MemoryToolResultStore
from tests.contracts.registry_factory import build_test_tool_registry
from osc_agent.tools.git import (
    GitDiffInput,
    GitDiffTool,
    GitLogInput,
    GitStatusInput,
)
from osc_agent.workspaces.git_state import git_workspace_fingerprint


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    (root / "tracked.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)


def context(root: Path) -> tool_context:
    return tool_context(session_id="session-1", working_directory=str(root), state_directory=str(root / "state"))


def test_git_tools_have_strict_contracts_and_read_only_concurrency() -> None:
    registry = build_test_tool_registry()
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
    executor = ToolExecutor(build_test_tool_registry())

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
    assert "-before" in diff.data["patch"]
    assert "+after" in diff.data["patch"]
    assert diff.data["files"][0]["path"] == "tracked.txt"
    assert "initial" in log.data["log"]


async def _gather(*calls):
    return await asyncio.gather(*calls)


def test_git_log_rejects_implicit_and_out_of_range_limits(tmp_path: Path) -> None:
    executor = ToolExecutor(build_test_tool_registry())

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
        ToolExecutor(build_test_tool_registry()).execute(
            ToolUseBlock(id="status-1", name="git_status", input={}),
            context(isolated),
        )
    )

    assert result.error and result.error.code == "GIT_COMMAND_FAILED"


def test_git_snapshot_includes_commits_staged_rename_and_untracked(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "committed.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "committed.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "ahead"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "mv", "tracked.txt", "renamed.txt"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.txt").write_text("new text\n", encoding="utf-8")
    tool_context = context(tmp_path)
    tool_context = tool_context.model_copy(
        update={"workspace": tool_context.workspace.model_copy(update={
            "worktree": WorktreeSession(
                path=str(tmp_path),
                original_working_directory=str(tmp_path),
                branch="test",
                base_commit=base,
            )
        })}
    )

    result = asyncio.run(
        ToolExecutor(build_test_tool_registry()).execute(
            ToolUseBlock(id="snapshot", name="git_diff", input={}),
            tool_context,
        )
    )

    assert result.error is None
    assert result.data["base_commit"] == base
    assert result.data["commits_ahead"] == 1
    paths = {item["path"]: item for item in result.data["files"]}
    assert paths["committed.txt"]["untracked"] is False
    assert paths["renamed.txt"]["staged"] is True
    assert paths["untracked.txt"]["untracked"] is True
    assert "b/untracked.txt" in result.data["patch"]
    assert "committed.txt" in result.data["patch"]
    assert result.data["has_changes"] is True
    assert result.data["workspace_fingerprint"]


def test_large_git_snapshot_is_persisted_with_session_ownership(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    (tmp_path / "large.txt").write_text("line\n" * 15_000, encoding="utf-8")
    store = MemoryToolResultStore()
    executor = ToolExecutor(ToolRegistry([GitDiffTool(store)]))

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="snapshot", name="git_diff", input={}),
            context(tmp_path),
        )
    )

    assert result.error is None
    assert result.data["truncated"] is True
    result_id = result.data["full_result_id"]
    assert result_id
    assert "b/large.txt" in store.read(
        session_id="session-1",
        result_id=result_id,
    )


def test_workspace_fingerprint_covers_git_visible_changes_and_ignores_cache(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    baseline = git_workspace_fingerprint(repo_root=tmp_path)

    (tmp_path / ".gitignore").write_text(".cache/\n", encoding="utf-8")
    ignored_baseline = git_workspace_fingerprint(repo_root=tmp_path)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "result.bin").write_bytes(b"\x00\x01")
    assert git_workspace_fingerprint(repo_root=tmp_path) == ignored_baseline

    (tmp_path / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
    unstaged = git_workspace_fingerprint(repo_root=tmp_path)
    assert unstaged != baseline
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    staged = git_workspace_fingerprint(repo_root=tmp_path)
    assert staged != unstaged

    binary = tmp_path / "untracked.bin"
    binary.write_bytes(b"\x00\x01")
    first_binary = git_workspace_fingerprint(repo_root=tmp_path)
    binary.write_bytes(b"\x00\x02")
    assert git_workspace_fingerprint(repo_root=tmp_path) != first_binary

    subprocess.run(["git", "add", ".gitignore", "tracked.txt", "untracked.bin"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "changed"], cwd=tmp_path, check=True)
    assert git_workspace_fingerprint(repo_root=tmp_path) != staged

    committed = git_workspace_fingerprint(repo_root=tmp_path)
    subprocess.run(["git", "mv", "tracked.txt", "renamed.txt"], cwd=tmp_path, check=True)
    assert git_workspace_fingerprint(repo_root=tmp_path) != committed
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").unlink()
    assert git_workspace_fingerprint(repo_root=tmp_path) != committed


def test_workspace_fingerprint_detects_dirty_submodule(tmp_path: Path) -> None:
    if shutil.which("basename") is None or shutil.which("sed") is None:
        pytest.skip("Git submodule shell helpers are unavailable in this Windows test environment")
    child = tmp_path / "child"
    parent = tmp_path / "parent"
    child.mkdir()
    parent.mkdir()
    initialize_repository(child)
    initialize_repository(parent)
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "--quiet",
            str(child),
            "vendor/child",
        ],
        cwd=parent,
        check=True,
    )
    subprocess.run(["git", "commit", "--quiet", "-am", "add submodule"], cwd=parent, check=True)
    baseline = git_workspace_fingerprint(repo_root=parent)

    (parent / "vendor" / "child" / "tracked.txt").write_text(
        "dirty\n",
        encoding="utf-8",
    )

    assert git_workspace_fingerprint(repo_root=parent) != baseline
