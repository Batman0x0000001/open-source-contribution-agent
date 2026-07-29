"""验证测试工具注册表工厂的契约、边界条件与回归行为。"""

from __future__ import annotations

from pathlib import Path
import tempfile

from osc_agent.workspaces.git_worktree import GitWorktreeManager
from osc_agent.runtime.context import MemoryToolResultStore
from osc_agent.tools.registry import build_tool_registry


def build_test_tool_registry(root: Path | None = None):
    state = (root or Path(tempfile.gettempdir()) / "osc-agent-contract-tests") / "state"
    return build_tool_registry(
        worktree_manager=GitWorktreeManager(state / "worktrees"),
        tool_result_store=MemoryToolResultStore(),
    )
