from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess

from pydantic import Field

from osc_agent.runtime.models import ContractModel, ToolError, ToolResult, ToolUseContext
from osc_agent.runtime.tool import BaseTool


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

    async def call(self, input: GitStatusInput, context: ToolUseContext) -> ToolResult:
        return _git_tool_result(
            "status",
            await asyncio.to_thread(git_status, repo_root=Path(context.working_directory)),
        )


class GitDiffInput(ContractModel):
    pass


class GitDiffOutput(ContractModel):
    diff: str


class GitDiffTool(BaseTool[GitDiffInput, GitDiffOutput]):
    name = "git_diff"
    description = "Show tracked Git changes in the current repository."
    input_model = GitDiffInput
    output_model = GitDiffOutput

    def is_read_only(self, input: GitDiffInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitDiffInput) -> bool:
        return True

    async def call(self, input: GitDiffInput, context: ToolUseContext) -> ToolResult:
        return _git_tool_result(
            "diff",
            await asyncio.to_thread(git_diff, repo_root=Path(context.working_directory)),
        )


class GitLogInput(ContractModel):
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of recent commits to return.")


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

    async def call(self, input: GitLogInput, context: ToolUseContext) -> ToolResult:
        return _git_tool_result(
            "log",
            await asyncio.to_thread(
                git_log,
                repo_root=Path(context.working_directory),
                limit=input.limit,
            ),
        )


def _git_tool_result(field: str, output: str) -> ToolResult:
    if output.startswith("Error: "):
        return ToolResult(
            error=ToolError(code="GIT_COMMAND_FAILED", message=output.removeprefix("Error: "))
        )
    return ToolResult(data={field: output})


def _run_git(repo_root: Path, arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Error: {exc}"
    stdout = (completed.stdout or "").strip("\r\n")
    if completed.returncode != 0:
        detail = "\n".join(
            part for part in (stdout, (completed.stderr or "").strip("\r\n")) if part
        )
        return f"Error: git command failed with exit code {completed.returncode}: {detail}"[:MAX_GIT_OUTPUT_CHARS]
    return (stdout or "(no output)")[:MAX_GIT_OUTPUT_CHARS]


def git_status(*, repo_root: Path) -> str:
    return _run_git(repo_root, ["status", "--short"])


def git_diff(*, repo_root: Path) -> str:
    return _run_git(repo_root, ["diff", "--no-ext-diff", "--no-textconv", "--"])


def git_log(*, repo_root: Path, limit: int = 5) -> str:
    return _run_git(repo_root, ["log", f"-{min(max(limit, 1), 50)}", "--oneline"])
