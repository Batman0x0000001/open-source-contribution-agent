"""检测声明式只读子 Agent 对 Git 工作区造成的可见变更。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.models import ToolError, ToolResult, ToolUseContext
from osc_agent.tools.git import git_workspace_fingerprint


async def capture_workspace_fingerprint(context: ToolUseContext) -> str | ToolResult:
    try:
        return await asyncio.to_thread(
            git_workspace_fingerprint,
            repo_root=Path(context.working_directory),
        )
    except (OSError, ValueError) as exc:
        return _error(
            "AGENT_READ_ONLY_GUARD_FAILED",
            f"unable to verify read-only Agent workspace: {exc}",
        )


async def verify_workspace_unchanged(
    before: str | ToolResult | None,
    context: ToolUseContext,
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
