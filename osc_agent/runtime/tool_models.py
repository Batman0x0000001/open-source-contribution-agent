"""定义 Tool 执行、权限和受控上下文更新契约。"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue

from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.state import StateChange


class ToolError(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class ToolResult(ContractModel):
    data: JsonValue = None
    new_messages: list[RuntimeMessage] = Field(default_factory=list)
    state_changes: tuple[StateChange, ...] = ()
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


PermissionDecision: TypeAlias = Annotated[
    Allow | Deny | Ask,
    Field(discriminator="decision"),
]
