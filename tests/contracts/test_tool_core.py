"""验证工具核心协议与执行的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio

from pydantic import Field

from osc_agent.runtime.hooks import HookBlock, HookContinue, HookRegistry
from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import (
    Ask,
    ApprovalResponse,
    CapabilityScope,
    Deny,
    ToolResult,
    ToolUseContext,
    ValidationFailure,
)
from osc_agent.runtime.tool import BaseTool, ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor


class EchoInput(ContractModel):
    value: str = Field(min_length=1)


class EchoOutput(ContractModel):
    value: str


class EchoTool(BaseTool[EchoInput, EchoOutput]):
    name = "echo"
    input_model = EchoInput
    output_model = EchoOutput

    def __init__(self) -> None:
        self.calls = 0

    def is_read_only(self, input: EchoInput) -> bool:
        return True

    def is_concurrency_safe(self, input: EchoInput) -> bool:
        return True

    async def call(self, input: EchoInput, context: ToolUseContext) -> ToolResult:
        self.calls += 1
        return ToolResult(data={"value": input.value})


class InvalidOutputTool(EchoTool):
    name = "invalid_output"

    async def call(self, input: EchoInput, context: ToolUseContext) -> ToolResult:
        return ToolResult(data={"value": 123})


class DestructiveTool(EchoTool):
    name = "destructive"

    def is_destructive(self, input: EchoInput) -> bool:
        return True


class ProcessTool(DestructiveTool):
    name = "process"

    def is_read_only(self, input: EchoInput) -> bool:
        return False

    def permission_risk(self, input: EchoInput) -> str:
        return "process"


class DestructiveDeniedByTool(DestructiveTool):
    name = "destructive_denied"

    async def check_permissions(self, input: EchoInput, context: ToolUseContext):
        return Deny(reason="tool-specific denial")


class RejectedTool(EchoTool):
    name = "rejected"

    async def validate_input(self, input: EchoInput, context: ToolUseContext):
        return ValidationFailure(reason="domain validation failed")


def context(*, allowed_tools: frozenset[str] | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="session-1",
        working_directory="C:/repo",
        state_directory="C:/state",
        capabilities=CapabilityScope(allowed_tools=allowed_tools),
    )


def test_executor_validates_input_and_output() -> None:
    echo = EchoTool()
    executor = ToolExecutor(ToolRegistry([echo, InvalidOutputTool()]))

    invalid_input = asyncio.run(executor.execute(
        ToolUseBlock(id="1", name="echo", input={"value": 1}),
        context(),
    ))
    invalid_output = asyncio.run(executor.execute(
        ToolUseBlock(id="2", name="invalid_output", input={"value": "ok"}),
        context(),
    ))

    assert invalid_input.error and invalid_input.error.code == "TOOL_INPUT_INVALID"
    assert invalid_output.error and invalid_output.error.code == "TOOL_OUTPUT_INVALID"
    assert echo.calls == 0


def test_capability_denial_prevents_side_effect() -> None:
    echo = EchoTool()
    executor = ToolExecutor(ToolRegistry([echo]))

    result = asyncio.run(executor.execute(
        ToolUseBlock(id="1", name="echo", input={"value": "blocked"}),
        context(allowed_tools=frozenset()),
    ))

    assert result.error and result.error.code == "PERMISSION_DENIED"
    assert echo.calls == 0


def test_destructive_tool_requires_explicit_approval() -> None:
    tool = DestructiveTool()
    denied = ToolExecutor(ToolRegistry([tool]))

    denied_result = asyncio.run(denied.execute(
        ToolUseBlock(id="1", name="destructive", input={"value": "x"}),
        context(),
    ))

    async def approve(decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    approved = ToolExecutor(
        ToolRegistry([tool]),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    approved_result = asyncio.run(approved.execute(
        ToolUseBlock(id="2", name="destructive", input={"value": "x"}),
        context(),
    ))

    assert denied_result.error and denied_result.error.code == "PERMISSION_REQUIRED"
    assert approved_result.error is None
    assert tool.calls == 1


def test_general_approval_does_not_bypass_tool_specific_permission() -> None:
    tool = DestructiveDeniedByTool()

    async def approve(decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    executor = ToolExecutor(
        ToolRegistry([tool]),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    result = asyncio.run(executor.execute(
        ToolUseBlock(id="1", name=tool.name, input={"value": "x"}),
        context(),
    ))

    assert result.error and result.error.code == "PERMISSION_DENIED"
    assert result.error.message == "tool-specific denial"
    assert tool.calls == 0


def test_session_permission_matches_exact_validated_input_and_cwd() -> None:
    approvals = 0

    async def approve(_decision: Ask) -> ApprovalResponse:
        nonlocal approvals
        approvals += 1
        return ApprovalResponse(choice="allow_for_session")

    tool = ProcessTool()
    executor = ToolExecutor(
        ToolRegistry([tool]),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    tool_context = context()

    for call_id in ("first", "second"):
        result = asyncio.run(
            executor.execute(
                ToolUseBlock(id=call_id, name=tool.name, input={"value": "same"}),
                tool_context,
            )
        )
        assert result.error is None
    changed = asyncio.run(
        executor.execute(
            ToolUseBlock(id="changed", name=tool.name, input={"value": "different"}),
            tool_context,
        )
    )

    assert changed.error is None
    assert approvals == 2
    assert len(tool_context.permission_grants) == 2


def test_cached_session_permission_cannot_bypass_plan_mode() -> None:
    async def approve(_decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_for_session")

    tool = ProcessTool()
    executor = ToolExecutor(
        ToolRegistry([tool]),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    tool_context = context()
    call = ToolUseBlock(id="first", name=tool.name, input={"value": "same"})
    assert asyncio.run(executor.execute(call, tool_context)).error is None
    tool_context.permission_mode = "plan"

    blocked = asyncio.run(executor.execute(call, tool_context))

    assert blocked.error and blocked.error.code == "PERMISSION_DENIED"


def test_validation_and_pre_hook_run_before_call() -> None:
    events: list[str] = []
    tool = RejectedTool()
    hooks = HookRegistry()

    async def pre_hook(payload, tool_context):
        events.append("pre")
        return HookContinue()

    hooks.register_pre_tool_use(pre_hook)
    executor = ToolExecutor(ToolRegistry([tool]), hooks=hooks)

    result = asyncio.run(executor.execute(
        ToolUseBlock(id="1", name="rejected", input={"value": "x"}),
        context(),
    ))

    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert events == []
    assert tool.calls == 0


def test_pre_hook_can_block_and_post_hook_observes_success() -> None:
    events: list[str] = []
    tool = EchoTool()
    hooks = HookRegistry()

    async def blocking_hook(payload, tool_context):
        events.append("pre")
        return HookBlock(reason="blocked by hook")

    async def post_hook(payload, tool_context):
        events.append("post")

    hooks.register_pre_tool_use(blocking_hook)
    hooks.register_post_tool_use(post_hook)
    executor = ToolExecutor(ToolRegistry([tool]), hooks=hooks)

    result = asyncio.run(executor.execute(
        ToolUseBlock(id="1", name="echo", input={"value": "x"}),
        context(),
    ))

    assert result.error and result.error.code == "HOOK_BLOCKED"
    assert events == ["pre"]
    assert tool.calls == 0
