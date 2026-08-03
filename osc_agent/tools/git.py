"""将 Git 查询能力封装为模型可调用的 Runtime 工具。"""

from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path

from pydantic import Field

from osc_agent.contracts import ContractModel
from osc_agent.runtime.state import ToolContext
from osc_agent.runtime.tool_models import ToolError, ToolResult
from osc_agent.runtime.session_store import ToolResultStore
from osc_agent.runtime.tool import BaseTool
from osc_agent.workspaces.git_state import (
    git_log,
    git_snapshot,
    git_status,
    git_workspace_fingerprint,
)


MAX_GIT_OUTPUT_CHARS = 50_000


class GitStatusInput(ContractModel):
    pass


class GitStatusOutput(ContractModel):
    status: str


class GitStatusTool(BaseTool[GitStatusInput, GitStatusOutput]):
    name = "git_status"
    description = "Show the concise Git working-tree status for the current repository."
    input_model = GitStatusInput
    output_model = GitStatusOutput

    def is_read_only(self, input: GitStatusInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitStatusInput) -> bool:
        return True

    async def call(self, input: GitStatusInput, context: ToolContext) -> ToolResult:
        try:
            output = await asyncio.to_thread(
                git_status, repo_root=Path(context.workspace.working_directory)
            )
        except ValueError as exc:
            return _git_error(exc)
        return ToolResult(data={"status": output})


class GitDiffInput(ContractModel):
    pass


class GitChangedFile(ContractModel):
    path: str
    status: str
    staged: bool
    unstaged: bool
    untracked: bool


class GitDiffOutput(ContractModel):
    base_commit: str
    files: list[GitChangedFile]
    commits_ahead: int = Field(ge=0)
    stat: str
    patch: str
    truncated: bool
    full_result_id: str | None = None
    has_changes: bool
    workspace_fingerprint: str


class GitDiffTool(BaseTool[GitDiffInput, GitDiffOutput]):
    name = "git_diff"
    description = (
        "Show the contribution snapshot, including commits and tracked or untracked changes."
    )
    input_model = GitDiffInput
    output_model = GitDiffOutput

    def __init__(self, tool_result_store: ToolResultStore) -> None:
        self.tool_result_store = tool_result_store

    def is_read_only(self, input: GitDiffInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitDiffInput) -> bool:
        return True

    async def call(self, input: GitDiffInput, context: ToolContext) -> ToolResult:
        try:
            before = await asyncio.to_thread(
                git_workspace_fingerprint,
                repo_root=Path(context.workspace.working_directory),
            )
            snapshot = await asyncio.to_thread(
                git_snapshot,
                repo_root=Path(context.workspace.working_directory),
                base_commit=(
                    context.workspace.worktree.base_commit
                    if context.workspace.worktree
                    else None
                ),
            )
            after = await asyncio.to_thread(
                git_workspace_fingerprint,
                repo_root=Path(context.workspace.working_directory),
            )
            if before != after:
                raise ValueError("Git workspace changed while the snapshot was being generated")
        except ValueError as exc:
            return ToolResult(error=ToolError(code="GIT_COMMAND_FAILED", message=str(exc)))
        patch = snapshot["patch"]
        assert isinstance(patch, str)
        truncated = len(patch) > MAX_GIT_OUTPUT_CHARS
        result_id = None
        if truncated:
            result_id = self.tool_result_store.persist(
                session_id=context.session_id,
                tool_use_id=f"git-diff-{sha256(patch.encode('utf-8')).hexdigest()[:16]}",
                content=patch,
            )
        snapshot["patch"] = patch[:MAX_GIT_OUTPUT_CHARS]
        snapshot["truncated"] = truncated
        snapshot["full_result_id"] = result_id
        snapshot["workspace_fingerprint"] = after
        return ToolResult(data=snapshot)


class GitLogInput(ContractModel):
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of recent commits to return.",
    )


class GitLogOutput(ContractModel):
    log: str


class GitLogTool(BaseTool[GitLogInput, GitLogOutput]):
    name = "git_log"
    description = "Show a bounded list of recent Git commits."
    input_model = GitLogInput
    output_model = GitLogOutput

    def is_read_only(self, input: GitLogInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitLogInput) -> bool:
        return True

    async def call(self, input: GitLogInput, context: ToolContext) -> ToolResult:
        try:
            output = await asyncio.to_thread(
                git_log,
                repo_root=Path(context.workspace.working_directory),
                limit=input.limit,
            )
        except ValueError as exc:
            return _git_error(exc)
        return ToolResult(data={"log": output})


def _git_error(error: ValueError) -> ToolResult:
    return ToolResult(
        error=ToolError(
            code="GIT_COMMAND_FAILED",
            message=str(error),
        )
    )
