"""构建并注册 Runtime 的全部内置核心工具。"""

from __future__ import annotations

from osc_agent.runtime.tool import ToolRegistry
from osc_agent.tools.filesystem import EditFileTool, GlobTool, ReadFileTool, WriteFileTool
from osc_agent.tools.git import GitDiffTool, GitLogTool, GitStatusTool
from osc_agent.tools.github import GitHubGetIssueTool, GitHubListIssuesTool
from osc_agent.tools.bash import BashTool
from osc_agent.tools.plan import EnterPlanModeTool, ExitPlanModeTool, ReadPlanTool, WritePlanTool
from osc_agent.tools.questions import AskUserQuestionTool, QuestionHandler
from osc_agent.tools.worktree import EnterWorktreeTool, ExitWorktreeTool
from osc_agent.tools.state import ReadToolResultTool
from osc_agent.tools.search import GrepTool
from osc_agent.workspaces.git_worktree import GitWorktreeManager
from osc_agent.runtime.instructions import RepositoryInstructionResolver
from osc_agent.runtime.session_store import ToolResultStore
from osc_agent.processes.contracts import ProcessRunner


def build_core_tool_registry(
    *,
    worktree_manager: GitWorktreeManager,
    tool_result_store: ToolResultStore,
    instruction_resolver: RepositoryInstructionResolver | None = None,
    subprocess_env_allowlist: frozenset[str] = frozenset(),
    process_runner: ProcessRunner | None = None,
    question_handler: QuestionHandler | None = None,
) -> ToolRegistry:
    """新 Runtime 的唯一内置 Tool 注册入口。"""

    instructions = instruction_resolver or RepositoryInstructionResolver()
    return ToolRegistry(
        [
            ReadFileTool(instructions),
            WriteFileTool(instructions),
            EditFileTool(instructions),
            GlobTool(),
            GrepTool(instructions),
            BashTool(
                environment_allowlist=subprocess_env_allowlist,
                process_runner=process_runner,
            ),
            GitStatusTool(),
            GitDiffTool(tool_result_store),
            GitLogTool(),
            GitHubListIssuesTool(),
            GitHubGetIssueTool(),
            AskUserQuestionTool(question_handler),
            EnterPlanModeTool(),
            WritePlanTool(),
            ReadPlanTool(),
            ExitPlanModeTool(),
            EnterWorktreeTool(worktree_manager, instructions),
            ExitWorktreeTool(worktree_manager, instructions),
            ReadToolResultTool(tool_result_store),
        ]
    )
