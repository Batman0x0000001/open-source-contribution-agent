"""检测声明式只读子 Agent 对 Git 工作区造成的可见变更。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.state import ToolContext
from osc_agent.runtime.tool_models import ToolError, ToolResult
from osc_agent.workspaces.git_state import git_workspace_fingerprint


async def capture_workspace_fingerprint(context: ToolContext) -> str | ToolResult:
    try:
        return await asyncio.to_thread(
            git_workspace_fingerprint,
            repo_root=Path(context.workspace.working_directory),
        )
    except (OSError, ValueError) as exc:
        return _error(
            "AGENT_READ_ONLY_GUARD_FAILED",
            f"unable to verify read-only Agent workspace: {exc}",
        )


async def verify_workspace_unchanged(
    before: str | ToolResult | None,
    context: ToolContext,
) -> tuple[ToolResult | None, str | None]:
    after = await capture_workspace_fingerprint(context)
    if isinstance(after, ToolResult):
        return after, None
    if before != after:
        return (
            _error(
                "AGENT_READ_ONLY_VIOLATION",
                "read-only Agent changed Git-visible repository state; changes were preserved for inspection",
            ),
            after,
        )
    return None, after


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(error=ToolError(code=code, message=message))
