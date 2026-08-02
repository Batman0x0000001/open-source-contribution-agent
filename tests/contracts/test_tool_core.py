"""验证工具核心协议与执行的契约、边界条件与回归行为。"""

from __future__ import annotations

from osc_agent.runtime.state import CapabilityScope, PlanModeEntered

from tests.runtime_factories import agent_run_state, tool_context

import asyncio

from pydantic import Field
import pytest

from osc_agent.runtime.hooks import HookBlock, HookContinue, HookRegistry
from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import (
    Allow,
    Ask,
    ApprovalResponse,
    Deny,
    ToolResult,
    ValidationFailure,
)
from osc_agent.runtime.tool import BaseTool, ToolRegistry
from osc_agent.runtime.tool_execution import (
    PostToolUseHookError,
    ToolExecutionDependencies,
    ToolExecutor,
)


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

    async def call(self, input: EchoInput, context: tool_context) -> ToolResult:
        self.calls += 1
        return ToolResult(data={"value": input.value})


class InvalidOutputTool(EchoTool):
    name = "invalid_output"

    async def call(self, input: EchoInput, context: tool_context) -> ToolResult:
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

    async def check_permissions(self, input: EchoInput, context: tool_context):
        return Deny(reason="tool-specific denial")


class RejectedTool(EchoTool):
    name = "rejected"

    async def validate_input(self, input: EchoInput, context: tool_context):
        return ValidationFailure(reason="domain validation failed")


def context(*, allowed_tools: frozenset[str] | None = None) -> tool_context:
    return tool_context(
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
    class AllowEverythingPolicy:
        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, tool, input, context):
            self.calls += 1
            return Allow(updated_input=input.model_dump(mode="json"))

    echo = EchoTool()
    policy = AllowEverythingPolicy()
    executor = ToolExecutor(ToolRegistry([echo]), permission_policy=policy)

    result = asyncio.run(executor.execute(
        ToolUseBlock(id="1", name="echo", input={"value": "blocked"}),
        context(allowed_tools=frozenset()),
    ))

    assert result.error and result.error.code == "PERMISSION_DENIED"
    assert echo.calls == 0
    assert policy.calls == 0


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
    state = agent_run_state("C:/repo")

    for call_id in ("first", "second"):
        result = asyncio.run(
            executor.execute(
                ToolUseBlock(id=call_id, name=tool.name, input={"value": "same"}),
                state.tool_context(session_id="session-1", state_directory="C:/state"),
            )
        )
        state = state.apply_all(result.state_changes)
        assert result.error is None
    changed = asyncio.run(
        executor.execute(
            ToolUseBlock(id="changed", name=tool.name, input={"value": "different"}),
            state.tool_context(session_id="session-1", state_directory="C:/state"),
        )
    )
    state = state.apply_all(changed.state_changes)

    assert changed.error is None
    assert approvals == 2
    assert len(state.permissions.grants) == 2


def test_cached_session_permission_cannot_bypass_plan_mode() -> None:
    async def approve(_decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_for_session")

    tool = ProcessTool()
    executor = ToolExecutor(
        ToolRegistry([tool]),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    state = agent_run_state("C:/repo")
    call = ToolUseBlock(id="first", name=tool.name, input={"value": "same"})
    first = asyncio.run(
        executor.execute(
            call,
            state.tool_context(session_id="session-1", state_directory="C:/state"),
        )
    )
    assert first.error is None
    state = state.apply_all(first.state_changes).apply(PlanModeEntered())

    blocked = asyncio.run(
        executor.execute(
            call,
            state.tool_context(session_id="session-1", state_directory="C:/state"),
        )
    )

    assert blocked.error and blocked.error.code == "PERMISSION_DENIED"


def test_validation_and_pre_hook_run_before_call() -> None:
    events: list[str] = []
    tool = RejectedTool()
    hooks = HookRegistry()

    async def pre_hook(payload, context):
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

    async def blocking_hook(payload, context):
        events.append("pre")
        return HookBlock(reason="blocked by hook")

    async def post_hook(payload, context):
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


@pytest.mark.parametrize(
    "phase",
    ["validation", "permission_policy", "approval", "tool_permission", "pre_hook"],
)
def test_unexpected_pre_execution_failures_are_structured(phase: str) -> None:
    class PhaseTool(EchoTool):
        name = "phase"

        def is_destructive(self, input: EchoInput) -> bool:
            return phase == "approval"

        async def validate_input(self, input: EchoInput, context: tool_context):
            if phase == "validation":
                raise RuntimeError("validation exploded")
            return await super().validate_input(input, context)

        async def check_permissions(self, input: EchoInput, context: tool_context):
            if phase == "tool_permission":
                raise RuntimeError("tool permission exploded")
            return await super().check_permissions(input, context)

    class PhasePolicy:
        async def decide(self, tool, input, context):
            if phase == "permission_policy":
                raise RuntimeError("permission policy exploded")
            if phase == "approval":
                return Ask(
                    tool_name=tool.name,
                    prompt="approve",
                    working_directory=context.workspace.working_directory,
                    risk="destructive",
                )
            return Allow(updated_input=input.model_dump(mode="json"))

    async def approval_handler(_decision: Ask) -> ApprovalResponse:
        raise RuntimeError("approval exploded")

    hooks = HookRegistry()

    async def pre_hook(payload, context):
        if phase == "pre_hook":
            raise RuntimeError("pre hook exploded")
        return HookContinue()

    hooks.register_pre_tool_use(pre_hook)
    tool = PhaseTool()
    executor = ToolExecutor(
        ToolRegistry([tool]),
        permission_policy=PhasePolicy(),
        hooks=hooks,
        dependencies=ToolExecutionDependencies(approval_handler=approval_handler),
    )

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="phase-1", name=tool.name, input={"value": "x"}),
            context(),
        )
    )

    assert result.error and result.error.code == "TOOL_PIPELINE_FAILED"
    assert tool.calls == 0


def test_post_hook_failure_preserves_the_completed_tool_result() -> None:
    tool = EchoTool()
    hooks = HookRegistry()

    async def post_hook(payload, context) -> None:
        raise RuntimeError("post hook exploded")

    hooks.register_post_tool_use(post_hook)
    executor = ToolExecutor(ToolRegistry([tool]), hooks=hooks)

    with pytest.raises(PostToolUseHookError) as raised:
        asyncio.run(
            executor.execute(
                ToolUseBlock(id="post-1", name=tool.name, input={"value": "done"}),
                context(),
            )
        )

    assert tool.calls == 1
    assert raised.value.tool_use_id == "post-1"
    assert raised.value.result.error is None
    assert raised.value.result.data == {"value": "done"}


def test_pipeline_does_not_swallow_cancellation() -> None:
    hooks = HookRegistry()

    async def cancelled_pre_hook(payload, context):
        raise asyncio.CancelledError

    hooks.register_pre_tool_use(cancelled_pre_hook)
    executor = ToolExecutor(ToolRegistry([EchoTool()]), hooks=hooks)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            executor.execute(
                ToolUseBlock(id="cancel-1", name="echo", input={"value": "x"}),
                context(),
            )
        )
