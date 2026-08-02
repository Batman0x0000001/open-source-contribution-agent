"""验证工具调用调度的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio

from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import ToolResult
from osc_agent.runtime.state import PlanSaved
from tests.runtime_factories import agent_run_state
from osc_agent.runtime.tool import BaseTool, ToolRegistry
from osc_agent.runtime.tool_execution import PostToolUseHookError, ToolExecutor
from osc_agent.runtime.tool_orchestration import (
    ToolBatchStateCommitted,
    ToolResultAvailable,
    partition_tool_calls,
    run_tools,
)
from osc_agent.runtime.hooks import HookRegistry


class DelayInput(ContractModel):
    name: str
    delay: float
    concurrency_safe: bool = True


class DelayOutput(ContractModel):
    name: str


class DelayTool(BaseTool[DelayInput, DelayOutput]):
    name = "delay"
    input_model = DelayInput
    output_model = DelayOutput

    def __init__(self, completion_order: list[str]) -> None:
        self.completion_order = completion_order

    def is_read_only(self, input: DelayInput) -> bool:
        return input.concurrency_safe

    def is_concurrency_safe(self, input: DelayInput) -> bool:
        return input.concurrency_safe

    async def call(self, input: DelayInput, context: tool_context) -> ToolResult:
        await asyncio.sleep(input.delay)
        self.completion_order.append(input.name)
        return ToolResult(
            data={"name": input.name},
            state_changes=(PlanSaved(path=input.name),),
        )


def call(id: str, name: str, delay: float, *, safe: bool = True) -> ToolUseBlock:
    return ToolUseBlock(
        id=id,
        name="delay",
        input={"name": name, "delay": delay, "concurrency_safe": safe},
    )


def context() -> tool_context:
    return tool_context(
        session_id="session-1",
        working_directory="C:/repo",
        state_directory="C:/state",
    )


def test_partition_groups_only_consecutive_safe_calls() -> None:
    registry = ToolRegistry([DelayTool([])])

    batches = partition_tool_calls(
        [call("1", "a", 0), call("2", "b", 0), call("3", "w", 0, safe=False), call("4", "c", 0)],
        ToolExecutor(registry),
    )

    assert [(batch.concurrency_safe, len(batch.calls)) for batch in batches] == [
        (True, 2),
        (False, 1),
        (True, 1),
    ]


def test_concurrent_completion_is_streamed_but_context_updates_follow_call_order() -> None:
    completion_order: list[str] = []
    registry = ToolRegistry([DelayTool(completion_order)])
    executor = ToolExecutor(registry)
    async def collect_updates():
        return [
            update
            async for update in run_tools(
            [call("1", "first", 0.02), call("2", "second", 0)],
                executor=executor,
                state=agent_run_state("C:/repo"),
                session_id="session-1",
                state_directory="C:/state",
                transcript_messages=[],
        )
        ]

    updates = asyncio.run(collect_updates())

    result_updates = [
        update for update in updates if isinstance(update, ToolResultAvailable)
    ]
    committed = next(
        update for update in updates if isinstance(update, ToolBatchStateCommitted)
    )

    assert completion_order == ["second", "first"]
    assert [update.tool_use_id for update in result_updates] == ["2", "1"]
    assert committed.agent_state.permissions.plan_path == "second"


def test_non_safe_calls_execute_serially() -> None:
    completion_order: list[str] = []
    registry = ToolRegistry([DelayTool(completion_order)])
    executor = ToolExecutor(registry)

    async def collect_updates():
        return [
            update
            async for update in run_tools(
            [call("1", "first", 0.01, safe=False), call("2", "second", 0, safe=False)],
                executor=executor,
                state=agent_run_state("C:/repo"),
                session_id="session-1",
                state_directory="C:/state",
                transcript_messages=[],
        )
        ]

    updates = asyncio.run(collect_updates())

    assert completion_order == ["first", "second"]
    commits = [
        update for update in updates if isinstance(update, ToolBatchStateCommitted)
    ]
    assert [update.agent_state.permissions.plan_path for update in commits] == [
        "first",
        "second",
    ]


def test_concurrent_post_hook_failure_commits_all_started_tool_results() -> None:
    completion_order: list[str] = []
    hooks = HookRegistry()

    async def fail_second(payload, context) -> None:
        if payload.input["name"] == "second":
            raise RuntimeError("post hook failed")

    hooks.register_post_tool_use(fail_second)
    executor = ToolExecutor(
        ToolRegistry([DelayTool(completion_order)]),
        hooks=hooks,
    )

    async def collect_updates():
        updates = []
        with_error = None
        try:
            async for update in run_tools(
                [call("1", "first", 0.02), call("2", "second", 0)],
                executor=executor,
                state=agent_run_state("C:/repo"),
                session_id="session-1",
                state_directory="C:/state",
                transcript_messages=[],
            ):
                updates.append(update)
        except PostToolUseHookError as exc:
            with_error = exc
        return updates, with_error

    updates, error = asyncio.run(collect_updates())
    results = [
        update for update in updates if isinstance(update, ToolResultAvailable)
    ]
    committed = next(
        update for update in updates if isinstance(update, ToolBatchStateCommitted)
    )

    assert completion_order == ["second", "first"]
    assert [update.tool_use_id for update in results] == ["2", "1"]
    assert committed.agent_state.permissions.plan_path == "second"
    assert error is not None and error.tool_use_id == "2"
