"""验证工具调用调度的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio

from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import ContextUpdate, ToolResult, ToolUseContext
from osc_agent.runtime.tool import BaseTool, ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.runtime.tool_orchestration import partition_tool_calls, run_tools


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

    async def call(self, input: DelayInput, context: ToolUseContext) -> ToolResult:
        await asyncio.sleep(input.delay)
        self.completion_order.append(input.name)
        return ToolResult(
            data={"name": input.name},
            context_update=ContextUpdate(plan_path=input.name),
        )


def call(id: str, name: str, delay: float, *, safe: bool = True) -> ToolUseBlock:
    return ToolUseBlock(
        id=id,
        name="delay",
        input={"name": name, "delay": delay, "concurrency_safe": safe},
    )


def context() -> ToolUseContext:
    return ToolUseContext(
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
            context=context(),
        )
        ]

    updates = asyncio.run(collect_updates())

    result_updates = [update for update in updates if update.result is not None]
    final_context = updates[-1].context

    assert completion_order == ["second", "first"]
    assert [update.tool_use_id for update in result_updates] == ["2", "1"]
    assert final_context.plan_path == "second"


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
            context=context(),
        )
        ]

    updates = asyncio.run(collect_updates())

    assert completion_order == ["first", "second"]
    assert updates[-1].context.plan_path == "second"
