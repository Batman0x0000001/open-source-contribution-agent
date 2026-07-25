from __future__ import annotations

from osc_agent.runtime.tool import ToolRegistry
from osc_agent.tools.file_tools import EditFileTool, GlobTool, ReadFileTool, WriteFileTool
from osc_agent.tools.git import GitDiffTool, GitLogTool, GitStatusTool
from osc_agent.tools.github import GitHubGetIssueTool, GitHubListIssuesTool
from osc_agent.tools.shell import ShellTool
from osc_agent.tools.interaction import AskUserQuestionTool, EnterPlanModeTool, ExitPlanModeTool, ReadPlanTool, WritePlanTool
from osc_agent.tools.worktree_tools import EnterWorktreeTool, ExitWorktreeTool
from osc_agent.tools.state_tools import ReadToolResultTool
from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.session_store import ToolResultStore


def build_core_tool_registry(*, worktree_manager: WorktreeManager, tool_result_store: ToolResultStore) -> ToolRegistry:
    """新 Runtime 的唯一内置 Tool 注册入口。"""

    return ToolRegistry(
        [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GlobTool(),
            ShellTool(),
            GitStatusTool(),
            GitDiffTool(),
            GitLogTool(),
            GitHubListIssuesTool(),
            GitHubGetIssueTool(),
            AskUserQuestionTool(),
            EnterPlanModeTool(),
            WritePlanTool(),
            ReadPlanTool(),
            ExitPlanModeTool(),
            EnterWorktreeTool(worktree_manager),
            ExitWorktreeTool(worktree_manager),
            ReadToolResultTool(tool_result_store),
        ]
    )
