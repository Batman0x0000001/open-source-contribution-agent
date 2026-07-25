from __future__ import annotations

from pathlib import Path
import tempfile

from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.context import MemoryToolResultStore
from osc_agent.tools.core import build_core_tool_registry


def build_test_tool_registry(root: Path | None = None):
    state = (root or Path(tempfile.gettempdir()) / "osc-agent-contract-tests") / "state"
    return build_core_tool_registry(
        worktree_manager=WorktreeManager(state / "worktrees"),
        tool_result_store=MemoryToolResultStore(),
    )
