"""定义 Agent Runtime 向上层流式发送的事件和终态。"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import RuntimeMessage, ToolUseBlock
from osc_agent.runtime.tool_models import ToolResult


class Complete(ContractModel):
    kind: Literal["complete"] = "complete"
    reason: str = Field(min_length=1)


class Blocked(ContractModel):
    kind: Literal["blocked"] = "blocked"
    reason: str = Field(min_length=1)


class Failed(ContractModel):
    kind: Literal["failed"] = "failed"
    error_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    retryable: bool = False


class Cancelled(ContractModel):
    kind: Literal["cancelled"] = "cancelled"
    reason: str = Field(min_length=1)


class ModelRequestStarted(ContractModel):
    type: Literal["model_request_started"] = "model_request_started"
    session_id: str = Field(min_length=1)
    round_number: int = Field(ge=1)


class ModelRetryScheduled(ContractModel):
    type: Literal["model_retry_scheduled"] = "model_retry_scheduled"
    attempt: int = Field(ge=2)
    max_attempts: int = Field(ge=2)
    delay_seconds: float = Field(ge=0)
    error_code: str = Field(min_length=1)
    had_partial_output: bool = False


class AssistantDelta(ContractModel):
    type: Literal["assistant_delta"] = "assistant_delta"
    text: str


class AssistantMessageCompleted(ContractModel):
    type: Literal["assistant_message_completed"] = "assistant_message_completed"
    message: RuntimeMessage


class ToolRequested(ContractModel):
    type: Literal["tool_requested"] = "tool_requested"
    call: ToolUseBlock


class ToolCompleted(ContractModel):
    type: Literal["tool_completed"] = "tool_completed"
    tool_use_id: str = Field(min_length=1)
    result: ToolResult


class ContextCompacted(ContractModel):
    type: Literal["context_compacted"] = "context_compacted"
    before_messages: int = Field(ge=0)
    after_messages: int = Field(ge=0)
    reason: str = Field(min_length=1)


class RunCompleted(ContractModel):
    type: Literal["run_completed"] = "run_completed"
    transition: Complete


class RunStopped(ContractModel):
    type: Literal["run_stopped"] = "run_stopped"
    transition: Blocked | Failed | Cancelled


RuntimeEvent: TypeAlias = Annotated[
    ModelRequestStarted
    | ModelRetryScheduled
    | AssistantDelta
    | AssistantMessageCompleted
    | ToolRequested
    | ToolCompleted
    | ContextCompacted
    | RunCompleted
    | RunStopped,
    Field(discriminator="type"),
]
