"""检测声明式只读子 Agent 对 Git 工作区造成的可见变更。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.workspaces.git_state import git_workspace_fingerprint


class SubagentWorkspaceError(RuntimeError):
    def __init__(self, *, code: str, message: str, fingerprint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.fingerprint = fingerprint


async def capture_workspace_fingerprint(working_directory: str) -> str:
    try:
        return await asyncio.to_thread(
            git_workspace_fingerprint,
            repo_root=Path(working_directory),
        )
    except (OSError, ValueError) as exc:
        raise SubagentWorkspaceError(
            code="AGENT_READ_ONLY_GUARD_FAILED",
            message=f"unable to verify read-only Agent workspace: {exc}",
        ) from exc


async def verify_workspace_unchanged(
    before: str,
    working_directory: str,
) -> str:
    after = await capture_workspace_fingerprint(working_directory)
    if before != after:
        raise SubagentWorkspaceError(
            code="AGENT_READ_ONLY_VIOLATION",
            message=(
                "read-only Agent changed Git-visible repository state; "
                "changes were preserved for inspection"
            ),
            fingerprint=after,
        )
    return after
