"""验证worktree 交互工具的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import apply_tool_result, tool_context

import asyncio
from pathlib import Path

from osc_agent.workspaces.git_worktree import GitWorktreeManager
from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import ApprovalResponse, Ask
from osc_agent.workspaces.models import WorktreeSession
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor
from tests.contracts.registry_factory import build_test_tool_registry


def context(root: Path) -> tool_context:
    return tool_context(session_id="session-1", working_directory=str(root), state_directory=str(root / "state"))


def test_worktree_name_is_validated_before_approval(tmp_path: Path) -> None:
    approvals = 0

    async def approve(decision: Ask) -> ApprovalResponse:
        nonlocal approvals
        approvals += 1
        return ApprovalResponse(choice="allow_once")

    result = asyncio.run(
        ToolExecutor(
            build_test_tool_registry(),
            dependencies=ToolExecutionDependencies(approval_handler=approve),
        ).execute(
            ToolUseBlock(id="create", name="enter_worktree", input={"name": "../escape"}),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert approvals == 0


def test_worktree_mutations_use_permission_pipeline(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / ".osc_agent" / "worktrees" / "feature"
    monkeypatch.setattr(
        GitWorktreeManager,
        "create",
        lambda self, repo_root, name: WorktreeSession(
            path=str(target), original_working_directory=str(tmp_path), branch="osc-agent/feature", base_commit="abc123"
        ),
    )
    call = ToolUseBlock(id="create", name="enter_worktree", input={"name": "feature"})
    registry = build_test_tool_registry(tmp_path)
    denied = asyncio.run(ToolExecutor(registry).execute(call, context(tmp_path)))

    async def approve(decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    approved = asyncio.run(
        ToolExecutor(
            registry,
            dependencies=ToolExecutionDependencies(approval_handler=approve),
        ).execute(call, context(tmp_path))
    )

    assert denied.error and denied.error.code == "PERMISSION_REQUIRED"
    assert approved.error is None
    assert approved.data["action"] == "entered"
    assert apply_tool_result(context(tmp_path), approved).workspace.worktree.path == str(target)
