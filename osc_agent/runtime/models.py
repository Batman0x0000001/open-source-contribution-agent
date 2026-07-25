from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class ContractModel(BaseModel):
    """跨模块数据的严格运行时契约。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class FrozenContractModel(BaseModel):
    """一次运行内不可变的严格契约。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )


class TextBlock(ContractModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(ContractModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResultBlock(ContractModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(min_length=1)
    content: JsonValue
    is_error: bool = False


MessageBlock: TypeAlias = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class RuntimeMessage(ContractModel):
    role: Literal["user", "assistant", "system"]
    content: list[MessageBlock] = Field(min_length=1)


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


class WorktreeSession(ContractModel):
    path: str = Field(min_length=1)
    original_working_directory: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)


class ContextUpdate(ContractModel):
    """Tool 声明的上下文修改，由 Runtime 统一应用。"""

    working_directory: str | None = None
    permission_mode: Literal["default", "plan"] | None = None
    plan_path: str | None = None
    clear_plan_path: bool = False
    worktree: WorktreeSession | None = None
    clear_worktree: bool = False
    capabilities: CapabilityScope | None = None


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
    prompt: str = Field(min_length=1)


PermissionDecision: TypeAlias = Annotated[
    Allow | Deny | Ask,
    Field(discriminator="decision"),
]


class ToolUseContext(ContractModel):
    session_id: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    repository_root: str = Field(min_length=1)
    state_directory: str = Field(min_length=1)
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    transcript_messages: list[RuntimeMessage] = Field(default_factory=list)
    permission_mode: Literal["default", "plan"] = "default"
    plan_path: str | None = None
    worktree: WorktreeSession | None = None


class ToolExecutionUpdate(ContractModel):
    tool_use_id: str | None = None
    result: ToolResult | None = None
    context: ToolUseContext

    @model_validator(mode="after")
    def validate_result_identity(self) -> "ToolExecutionUpdate":
        if (self.tool_use_id is None) != (self.result is None):
            raise ValueError("tool_use_id and result must either both be set or both be omitted")
        return self


class QueryConfig(FrozenContractModel):
    max_rounds: int = Field(default=30, ge=1)
    max_total_tokens: int = Field(default=200_000, ge=1)
    deadline_seconds: int = Field(default=1_800, ge=1)
    max_tool_result_chars: int = Field(default=20_000, ge=1)
    keep_recent_tool_results: int = Field(default=3, ge=0)
    auto_compact_chars: int = Field(default=120_000, ge=1)
    max_reactive_compactions: int = Field(default=1, ge=0)
    max_no_progress_rounds: int = Field(default=3, ge=1)
    max_output_tokens: int = Field(default=8_192, ge=1)


class StartQueryParams(FrozenContractModel):
    mode: Literal["start"] = "start"
    session_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt: str = ""
    messages: list[RuntimeMessage] = Field(default_factory=list)
    repository_root: str = Field(min_length=1)
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    config: QueryConfig = Field(default_factory=QueryConfig)
    @model_validator(mode="after")
    def start_requires_messages(self) -> "StartQueryParams":
        if not self.messages:
            raise ValueError("new query requires at least one message")
        return self


class ResumeQueryParams(FrozenContractModel):
    mode: Literal["resume"] = "resume"
    session_id: str = Field(min_length=1)
    repository_root: str = Field(min_length=1)
    messages: list[RuntimeMessage] = Field(default_factory=list)
    config: QueryConfig = Field(default_factory=QueryConfig)


QueryParams: TypeAlias = Annotated[
    StartQueryParams | ResumeQueryParams,
    Field(discriminator="mode"),
]


class SessionMetadata(FrozenContractModel):
    session_id: str = Field(min_length=1)
    repository_root: str = Field(min_length=1)
    initial_working_directory: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt: str = ""
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)


class SessionRuntimeState(ContractModel):
    permission_mode: Literal["default", "plan"] = "default"
    plan_path: str | None = None
    worktree: WorktreeSession | None = None
    capabilities: CapabilityScope | None = None


class SessionSnapshot(ContractModel):
    metadata: SessionMetadata
    messages: list[RuntimeMessage] = Field(default_factory=list)
    runtime_state: SessionRuntimeState = Field(default_factory=SessionRuntimeState)


class QueryState(ContractModel):
    session_id: str = Field(min_length=1)
    round_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reactive_compaction_count: int = Field(default=0, ge=0)
    no_progress_rounds: int = Field(default=0, ge=0)
    last_tool_signature: str | None = None
    status: Literal["running", "completed", "blocked", "failed", "cancelled"] = "running"
    stop_reason: str | None = None


class ContinueWithTools(ContractModel):
    kind: Literal["continue_with_tools"] = "continue_with_tools"
    tool_calls: list[ToolUseBlock] = Field(min_length=1)


class ContinueAfterCompact(ContractModel):
    kind: Literal["continue_after_compact"] = "continue_after_compact"
    reason: str = Field(min_length=1)


class ContinueAfterRecovery(ContractModel):
    kind: Literal["continue_after_recovery"] = "continue_after_recovery"
    reason: str = Field(min_length=1)
    attempt: int = Field(ge=1)


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


Transition: TypeAlias = Annotated[
    ContinueWithTools
    | ContinueAfterCompact
    | ContinueAfterRecovery
    | Complete
    | Blocked
    | Failed
    | Cancelled,
    Field(discriminator="kind"),
]


class ModelRequestStarted(ContractModel):
    type: Literal["model_request_started"] = "model_request_started"
    session_id: str = Field(min_length=1)
    round_number: int = Field(ge=1)


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
    | AssistantDelta
    | AssistantMessageCompleted
    | ToolRequested
    | ToolCompleted
    | ContextCompacted
    | RunCompleted
    | RunStopped,
    Field(discriminator="type"),
]
