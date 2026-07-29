"""对模型请求的多个工具调用进行分组和安全调度。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import (
    ContextUpdate,
    ToolExecutionUpdate,
    ToolResult,
    ToolUseContext,
)
from osc_agent.runtime.tool_execution import ToolExecutor


@dataclass(frozen=True)
class ToolBatch:
    concurrency_safe: bool
    calls: tuple[ToolUseBlock, ...]


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
    context: ToolUseContext,
) -> AsyncIterator[ToolExecutionUpdate]:
    current_context = context.model_copy(deep=True)
    for batch in partition_tool_calls(calls, executor):
        if batch.concurrency_safe:
            results: dict[int, ToolResult] = {}

            async def execute_indexed(index: int, call: ToolUseBlock) -> tuple[int, ToolResult]:
                isolated_context = current_context.model_copy(deep=True)
                return index, await executor.execute(call, isolated_context)

            tasks = [
                asyncio.create_task(execute_indexed(index, call))
                for index, call in enumerate(batch.calls)
            ]
            try:
                for completed in asyncio.as_completed(tasks):
                    index, result = await completed
                    results[index] = result
                    yield ToolExecutionUpdate(
                        tool_use_id=batch.calls[index].id,
                        result=result,
                        context=current_context,
                    )
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            for index in range(len(batch.calls)):
                current_context = _apply_context_update(current_context, results[index].context_update)
            yield ToolExecutionUpdate(context=current_context)
            continue

        call = batch.calls[0]
        result = await executor.execute(call, current_context)
        current_context = _apply_context_update(current_context, result.context_update)
        yield ToolExecutionUpdate(
            tool_use_id=call.id,
            result=result,
            context=current_context,
        )


def _is_concurrency_safe(call: ToolUseBlock, executor: ToolExecutor) -> bool:
    tool = executor.registry.get(call.name)
    if tool is None:
        return False
    try:
        parsed = tool.input_model.model_validate(call.input)
        return bool(tool.is_concurrency_safe(parsed))
    except Exception:  # noqa: BLE001 - 判断失败时必须保守地按不可并发处理。
        return False


def _apply_context_update(
    context: ToolUseContext,
    update: ContextUpdate | None,
) -> ToolUseContext:
    if update is None:
        return context
    values = {}
    if update.working_directory is not None:
        values["working_directory"] = update.working_directory
    if update.permission_mode is not None:
        values["permission_mode"] = update.permission_mode
    if update.plan_path is not None or update.clear_plan_path:
        values["plan_path"] = update.plan_path
    if update.worktree is not None or update.clear_worktree:
        values["worktree"] = update.worktree
    if update.capabilities is not None:
        values["capabilities"] = context.capabilities.intersect(update.capabilities)
    if update.instruction_state is not None:
        values["instruction_state"] = (
            update.instruction_state
            if update.replace_instruction_state
            else type(update.instruction_state)(
                active_paths=sorted(
                    set(context.instruction_state.active_paths)
                    | set(update.instruction_state.active_paths)
                )
            )
        )
    if update.file_observations is not None:
        values["file_observations"] = (
            update.file_observations
            if update.replace_file_observations
            else {**context.file_observations, **update.file_observations}
        )
    elif update.replace_file_observations:
        values["file_observations"] = {}
    if update.completion_requirements is not None:
        values["completion_requirements"] = context.completion_requirements.tighten(
            update.completion_requirements
        )
    return context.model_copy(update=values, deep=True)
