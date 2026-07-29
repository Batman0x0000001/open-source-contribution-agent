"""定义 Tool 执行、权限和受控上下文更新契约。"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, model_validator

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


class ContextUpdate(ContractModel):
    """Tool 声明的上下文修改，由 Runtime 统一应用。"""

    working_directory: str | None = None
    permission_mode: Literal["default", "plan"] | None = None
    plan_path: str | None = None
    clear_plan_path: bool = False
    worktree: WorktreeSession | None = None
    clear_worktree: bool = False
    capabilities: CapabilityScope | None = None
    instruction_state: RepositoryInstructionState | None = None
    replace_instruction_state: bool = False
    file_observations: dict[str, FileObservation] | None = None
    replace_file_observations: bool = False
    completion_requirements: CompletionRequirements | None = None


class ToolError(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class ToolResult(ContractModel):
    data: JsonValue = None
    new_messages: list[RuntimeMessage] = Field(default_factory=list)
    context_update: ContextUpdate | None = None
    error: ToolError | None = None


class ValidationSuccess(ContractModel):
    valid: Literal[True] = True


class ValidationFailure(ContractModel):
    valid: Literal[False] = False
    reason: str = Field(min_length=1)


ValidationResult: TypeAlias = Annotated[
    ValidationSuccess | ValidationFailure,
    Field(discriminator="valid"),
]


class Allow(ContractModel):
    decision: Literal["allow"] = "allow"
    updated_input: dict[str, JsonValue] = Field(default_factory=dict)


class Deny(ContractModel):
    decision: Literal["deny"] = "deny"
    reason: str = Field(min_length=1)


class Ask(ContractModel):
    decision: Literal["ask"] = "ask"
    tool_name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    risk: Literal["write", "process", "destructive", "external"]
    preview: dict[str, JsonValue] = Field(default_factory=dict)


class ApprovalResponse(FrozenContractModel):
    choice: Literal["allow_once", "allow_for_session", "deny"]


class PermissionGrant(FrozenContractModel):
    """Session 内可复用的精确授权；只保存不可逆 fingerprint。"""

    tool_name: str = Field(min_length=1)
    risk: Literal["write", "process"]
    working_directory: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=64, max_length=64)


PermissionDecision: TypeAlias = Annotated[
    Allow | Deny | Ask,
    Field(discriminator="decision"),
]


class ToolUseContext(ContractModel):
    session_id: str = Field(min_length=1)
    tool_use_id: str | None = None
    working_directory: str = Field(min_length=1)
    state_directory: str = Field(min_length=1)
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    transcript_messages: list[RuntimeMessage] = Field(default_factory=list)
    permission_mode: Literal["default", "plan"] = "default"
    plan_path: str | None = None
    worktree: WorktreeSession | None = None
    instruction_state: RepositoryInstructionState = Field(default_factory=RepositoryInstructionState)
    file_observations: dict[str, FileObservation] = Field(default_factory=dict)
    completion_requirements: CompletionRequirements = Field(default_factory=CompletionRequirements)
    permission_grants: list[PermissionGrant] = Field(default_factory=list)


class ToolExecutionUpdate(ContractModel):
    tool_use_id: str | None = None
    result: ToolResult | None = None
    context: ToolUseContext

    @model_validator(mode="after")
    def validate_result_identity(self) -> "ToolExecutionUpdate":
        if (self.tool_use_id is None) != (self.result is None):
            raise ValueError("tool_use_id and result must either both be set or both be omitted")
        return self
