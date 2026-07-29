"""为 Runtime 契约测试构建领域化的 ToolContext。"""

from __future__ import annotations

from osc_agent.completion.models import CompletionRequirements
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.state import (
    AgentRunState,
    CapabilityScope,
    PermissionGrant,
    PermissionState,
    ToolContext,
    WorkspaceState,
)
from osc_agent.runtime.tool_models import ToolResult
from osc_agent.workspaces.models import FileObservation, RepositoryInstructionState, WorktreeSession


def tool_context(
    *,
    session_id: str,
    working_directory: str,
    state_directory: str,
    tool_use_id: str | None = None,
    transcript_messages: list[RuntimeMessage] | tuple[RuntimeMessage, ...] = (),
    worktree: WorktreeSession | None = None,
    instruction_state: RepositoryInstructionState | None = None,
    file_observations: dict[str, FileObservation] | None = None,
    permission_mode: str = "default",
    plan_path: str | None = None,
    permission_grants: tuple[PermissionGrant, ...] = (),
    capabilities: CapabilityScope | None = None,
    completion_requirements: CompletionRequirements | None = None,
) -> ToolContext:
    return ToolContext(
        session_id=session_id,
        tool_use_id=tool_use_id,
        state_directory=state_directory,
        transcript_messages=tuple(transcript_messages),
        workspace=WorkspaceState(
            working_directory=working_directory,
            worktree=worktree,
            instruction_state=instruction_state or RepositoryInstructionState(),
            file_observations=file_observations or {},
        ),
        permissions=PermissionState(
            mode=permission_mode,  # type: ignore[arg-type]
            plan_path=plan_path,
            grants=permission_grants,
        ),
        capabilities=capabilities or CapabilityScope(),
        completion_requirements=completion_requirements or CompletionRequirements(),
    )


def agent_run_state(
    workspace_root: str,
    *,
    capabilities: CapabilityScope | None = None,
    completion_requirements: CompletionRequirements | None = None,
    status: str | None = None,
) -> AgentRunState:
    state = AgentRunState.start(
        workspace_root=workspace_root,
        capabilities=capabilities or CapabilityScope(),
        completion_requirements=completion_requirements or CompletionRequirements(),
        instruction_state=RepositoryInstructionState(),
    )
    if status is not None:
        state = state.with_status(status)  # type: ignore[arg-type]
    return state


def apply_tool_result(context: ToolContext, result: ToolResult) -> ToolContext:
    state = AgentRunState(
        workspace=context.workspace,
        permissions=context.permissions,
        capabilities=context.capabilities,
        completion_requirements=context.completion_requirements,
    ).apply_all(result.state_changes)
    return state.tool_context(
        session_id=context.session_id,
        state_directory=context.state_directory,
        transcript_messages=context.transcript_messages,
        tool_use_id=context.tool_use_id,
    )
