"""按安全顺序调度 Tool，并将状态变化统一交给 AgentRunState。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, TypeAlias

from osc_agent.runtime.messages import RuntimeMessage, ToolUseBlock
from osc_agent.runtime.state import AgentRunState
from osc_agent.runtime.tool_execution import PostToolUseHookError, ToolExecutor
from osc_agent.runtime.tool_models import ToolResult


@dataclass(frozen=True)
class ToolBatch:
    concurrency_safe: bool
    calls: tuple[ToolUseBlock, ...]


@dataclass(frozen=True)
class ToolResultAvailable:
    tool_use_id: str
    result: ToolResult


@dataclass(frozen=True)
class ToolBatchStateCommitted:
    agent_state: AgentRunState


ToolExecutionEvent: TypeAlias = ToolResultAvailable | ToolBatchStateCommitted


def partition_tool_calls(
    calls: list[ToolUseBlock],
    executor: ToolExecutor,
) -> list[ToolBatch]:
    batches: list[ToolBatch] = []
    for call in calls:
        concurrency_safe = _is_concurrency_safe(call, executor)
        if concurrency_safe and batches and batches[-1].concurrency_safe:
            previous = batches[-1]
            batches[-1] = ToolBatch(True, (*previous.calls, call))
        else:
            batches.append(ToolBatch(concurrency_safe, (call,)))
    return batches


async def run_tools(
    calls: list[ToolUseBlock],
    *,
    executor: ToolExecutor,
    state: AgentRunState,
    session_id: str,
    state_directory: str,
    transcript_messages: list[RuntimeMessage],
) -> AsyncIterator[ToolExecutionEvent]:
    current_state = state
    for batch in partition_tool_calls(calls, executor):
        if batch.concurrency_safe:
            results: dict[int, ToolResult] = {}
            post_hook_failures: dict[int, PostToolUseHookError] = {}

            async def execute_indexed(
                index: int,
                call: ToolUseBlock,
            ) -> tuple[int, ToolResult, PostToolUseHookError | None]:
                context = current_state.tool_context(
                    session_id=session_id,
                    tool_use_id=call.id,
                    state_directory=state_directory,
                    transcript_messages=transcript_messages,
                )
                try:
                    return index, await executor.execute(call, context), None
                except PostToolUseHookError as exc:
                    return index, exc.result, exc

            tasks = [
                asyncio.create_task(execute_indexed(index, call))
                for index, call in enumerate(batch.calls)
            ]
            try:
                for completed in asyncio.as_completed(tasks):
                    index, result, post_hook_failure = await completed
                    results[index] = result
                    if post_hook_failure is not None:
                        post_hook_failures[index] = post_hook_failure
                    yield ToolResultAvailable(
                        tool_use_id=batch.calls[index].id,
                        result=result,
                    )
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            for index in range(len(batch.calls)):
                current_state = current_state.apply_all(results[index].state_changes)
            yield ToolBatchStateCommitted(agent_state=current_state)
            if post_hook_failures:
                raise post_hook_failures[min(post_hook_failures)]
            continue

        call = batch.calls[0]
        context = current_state.tool_context(
            session_id=session_id,
            tool_use_id=call.id,
            state_directory=state_directory,
            transcript_messages=transcript_messages,
        )
        post_hook_failure: PostToolUseHookError | None = None
        try:
            result = await executor.execute(call, context)
        except PostToolUseHookError as exc:
            result = exc.result
            post_hook_failure = exc
        current_state = current_state.apply_all(result.state_changes)
        yield ToolBatchStateCommitted(agent_state=current_state)
        yield ToolResultAvailable(
            tool_use_id=call.id,
            result=result,
        )
        if post_hook_failure is not None:
            raise post_hook_failure


def _is_concurrency_safe(call: ToolUseBlock, executor: ToolExecutor) -> bool:
    tool = executor.registry.get(call.name)
    if tool is None:
        return False
    try:
        parsed = tool.input_model.model_validate(call.input)
        return bool(tool.is_concurrency_safe(parsed))
    except Exception:  # noqa: BLE001 - 判断失败时必须保守地串行处理。
        return False
