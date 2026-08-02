"""定义一次 Agent 运行的唯一状态、Tool 只读视图和类型化状态变化。"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from osc_agent.completion.models import CompletionRequirements
from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.workspaces.models import (
    FileObservation,
    RepositoryInstructionState,
    WorktreeSession,
)


class CapabilityScope(FrozenContractModel):
    allowed_tools: frozenset[str] | None = None

    def permits_tool(self, tool_name: str) -> bool:
        return self.allowed_tools is None or tool_name in self.allowed_tools

    def intersect(self, other: "CapabilityScope") -> "CapabilityScope":
        if self.allowed_tools is None:
            return other
        if other.allowed_tools is None:
            return self
        return CapabilityScope(allowed_tools=self.allowed_tools & other.allowed_tools)


class PermissionGrant(FrozenContractModel):
    tool_name: str = Field(min_length=1)
    risk: Literal["write", "process"]
    working_directory: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=64, max_length=64)


class WorkspaceState(FrozenContractModel):
    working_directory: str = Field(min_length=1)
    worktree: WorktreeSession | None = None
    instruction_state: RepositoryInstructionState = Field(
        default_factory=RepositoryInstructionState
    )
    file_observations: dict[str, FileObservation] = Field(default_factory=dict)


class PermissionState(FrozenContractModel):
    mode: Literal["default", "plan"] = "default"
    plan_path: str | None = None
    grants: tuple[PermissionGrant, ...] = ()


class ToolContext(FrozenContractModel):
    """Runtime 为单次 Tool call 投影的只读上下文。"""

    session_id: str = Field(min_length=1)
    tool_use_id: str | None = None
    state_directory: str = Field(min_length=1)
    transcript_messages: tuple[RuntimeMessage, ...] = ()
    workspace: WorkspaceState
    permissions: PermissionState = Field(default_factory=PermissionState)
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    completion_requirements: CompletionRequirements = Field(
        default_factory=CompletionRequirements
    )


class WorktreeEntered(FrozenContractModel):
    kind: Literal["worktree_entered"] = "worktree_entered"
    session: WorktreeSession


class WorktreeExited(FrozenContractModel):
    kind: Literal["worktree_exited"] = "worktree_exited"
    working_directory: str = Field(min_length=1)


class InstructionsActivated(FrozenContractModel):
    kind: Literal["instructions_activated"] = "instructions_activated"
    state: RepositoryInstructionState
    replace: bool = False


class FilesObserved(FrozenContractModel):
    kind: Literal["files_observed"] = "files_observed"
    observations: dict[str, FileObservation] = Field(default_factory=dict)
    replace: bool = False


class PlanModeEntered(FrozenContractModel):
    kind: Literal["plan_mode_entered"] = "plan_mode_entered"


class PlanSaved(FrozenContractModel):
    kind: Literal["plan_saved"] = "plan_saved"
    path: str = Field(min_length=1)


class PlanModeExited(FrozenContractModel):
    kind: Literal["plan_mode_exited"] = "plan_mode_exited"


class CapabilitiesRestricted(FrozenContractModel):
    kind: Literal["capabilities_restricted"] = "capabilities_restricted"
    capabilities: CapabilityScope


class CompletionTightened(FrozenContractModel):
    kind: Literal["completion_tightened"] = "completion_tightened"
    requirements: CompletionRequirements


class PermissionGranted(FrozenContractModel):
    kind: Literal["permission_granted"] = "permission_granted"
    grant: PermissionGrant


StateChange: TypeAlias = Annotated[
    WorktreeEntered
    | WorktreeExited
    | InstructionsActivated
    | FilesObserved
    | PlanModeEntered
    | PlanSaved
    | PlanModeExited
    | CapabilitiesRestricted
    | CompletionTightened
    | PermissionGranted,
    Field(discriminator="kind"),
]


class AgentRunState(ContractModel):
    """Session V6 持久化并在 Query 各轮之间传递的唯一运行状态。"""

    workspace: WorkspaceState
    permissions: PermissionState = Field(default_factory=PermissionState)
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    completion_requirements: CompletionRequirements = Field(
        default_factory=CompletionRequirements
    )
    last_status: Literal[
        "running", "completed", "blocked", "failed", "cancelled"
    ] | None = None
    last_reason: str | None = None

    @classmethod
    def start(
        cls,
        *,
        workspace_root: str,
        capabilities: CapabilityScope,
        completion_requirements: CompletionRequirements,
        instruction_state: RepositoryInstructionState,
    ) -> "AgentRunState":
        return cls(
            workspace=WorkspaceState(
                working_directory=workspace_root,
                instruction_state=instruction_state,
            ),
            capabilities=capabilities,
            completion_requirements=completion_requirements,
        )

    def tool_context(
        self,
        *,
        session_id: str,
        state_directory: str,
        transcript_messages: list[RuntimeMessage] | tuple[RuntimeMessage, ...] = (),
        tool_use_id: str | None = None,
    ) -> ToolContext:
        return ToolContext(
            session_id=session_id,
            tool_use_id=tool_use_id,
            state_directory=state_directory,
            transcript_messages=tuple(transcript_messages),
            workspace=self.workspace,
            permissions=self.permissions,
            capabilities=self.capabilities,
            completion_requirements=self.completion_requirements,
        )

    def apply(self, change: StateChange) -> "AgentRunState":
        workspace = self.workspace
        permissions = self.permissions
        values: dict[str, object] = {}
        if isinstance(change, WorktreeEntered):
            workspace = workspace.model_copy(
                update={
                    "working_directory": change.session.path,
                    "worktree": change.session,
                },
                deep=True,
            )
        elif isinstance(change, WorktreeExited):
            workspace = workspace.model_copy(
                update={
                    "working_directory": change.working_directory,
                    "worktree": None,
                },
                deep=True,
            )
        elif isinstance(change, InstructionsActivated):
            instruction_state = (
                change.state
                if change.replace
                else RepositoryInstructionState(
                    active_paths=sorted(
                        set(workspace.instruction_state.active_paths)
                        | set(change.state.active_paths)
                    )
                )
            )
            workspace = workspace.model_copy(
                update={"instruction_state": instruction_state}, deep=True
            )
        elif isinstance(change, FilesObserved):
            observations = (
                dict(change.observations)
                if change.replace
                else {**workspace.file_observations, **change.observations}
            )
            workspace = workspace.model_copy(
                update={"file_observations": observations}, deep=True
            )
        elif isinstance(change, PlanModeEntered):
            permissions = permissions.model_copy(update={"mode": "plan"}, deep=True)
        elif isinstance(change, PlanSaved):
            permissions = permissions.model_copy(update={"plan_path": change.path}, deep=True)
        elif isinstance(change, PlanModeExited):
            permissions = permissions.model_copy(update={"mode": "default"}, deep=True)
        elif isinstance(change, CapabilitiesRestricted):
            values["capabilities"] = self.capabilities.intersect(change.capabilities)
        elif isinstance(change, CompletionTightened):
            values["completion_requirements"] = self.completion_requirements.tighten(
                change.requirements
            )
        elif isinstance(change, PermissionGranted):
            if change.grant not in permissions.grants:
                permissions = permissions.model_copy(
                    update={"grants": (*permissions.grants, change.grant)}, deep=True
                )
        values["workspace"] = workspace
        values["permissions"] = permissions
        return self.model_copy(update=values, deep=True)

    def apply_all(self, changes: tuple[StateChange, ...]) -> "AgentRunState":
        state = self
        for change in changes:
            state = state.apply(change)
        return state

    def with_status(
        self,
        status: Literal["running", "completed", "blocked", "failed", "cancelled"],
        reason: str | None = None,
    ) -> "AgentRunState":
        return self.model_copy(
            update={"last_status": status, "last_reason": reason}, deep=True
        )
