"""实现模型请求、工具执行、结果回填和停止判定的主循环。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Literal

from osc_agent.runtime.context import SessionTranscript
from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelCompleted, ModelGatewayError, ModelRequest, ModelTextDelta
from osc_agent.runtime.models import (
    AssistantDelta,
    AssistantMessageCompleted,
    Blocked,
    Cancelled,
    Complete,
    ContextCompacted,
    Failed,
    ModelRequestStarted,
    ModelRetryScheduled,
    QueryConfig,
    QueryParams,
    ResumeQueryParams,
    StartQueryParams,
    QueryState,
    RunCompleted,
    RunStopped,
    RuntimeEvent,
    RuntimeMessage,
    TextBlock,
    ToolCompleted,
    ToolError,
    ToolRequested,
    ToolResultBlock,
    ToolResult,
    ToolUseBlock,
    ToolUseContext,
    SessionMetadata,
    SessionRuntimeState,
)
from osc_agent.runtime.hooks import StopHookPayload
from osc_agent.runtime.tool_orchestration import run_tools


class AgentRuntime:
    def __init__(self, dependencies: QueryDependencies) -> None:
        self.dependencies = dependencies

    async def query(self, params: QueryParams) -> AsyncIterator[RuntimeEvent]:
        store = self.dependencies.session_store
        if store is None:
            async for event in self._query_locked(params):
                yield event
            return
        with store.lease(params.session_id):
            try:
                async for event in self._query_locked(params):
                    yield event
            except asyncio.CancelledError:
                snapshot = store.load(params.session_id)
                if snapshot is not None:
                    store.save_state(
                        params.session_id,
                        snapshot.runtime_state.model_copy(
                            update={
                                "last_status": "cancelled",
                                "last_reason": "query was cancelled",
                            },
                            deep=True,
                        ),
                    )
                raise

    async def _query_locked(self, params: QueryParams) -> AsyncIterator[RuntimeEvent]:
        state = QueryState(session_id=params.session_id)
        runtime_state = SessionRuntimeState()
        messages = list(params.messages)
        store = self.dependencies.session_store
        repository_root = str(Path(params.repository_root).resolve())
        if isinstance(params, ResumeQueryParams):
            if store is None:
                raise ValueError("resume requires a SessionStore")
            snapshot = store.load(params.session_id)
            if snapshot is None:
                raise ValueError(f"unknown session: {params.session_id}")
            if str(Path(snapshot.metadata.repository_root).resolve()) != repository_root:
                raise ValueError("resume repository does not match the saved session")
            messages = [*snapshot.messages, *messages]
            model = snapshot.metadata.model
            system_prompt = snapshot.metadata.system_prompt
            capabilities = snapshot.metadata.capabilities
            completion_requirements = snapshot.metadata.completion_requirements
            runtime_state = snapshot.runtime_state
            if runtime_state.capabilities is not None:
                capabilities = capabilities.intersect(runtime_state.capabilities)
            if runtime_state.completion_requirements is not None:
                completion_requirements = completion_requirements.tighten(
                    runtime_state.completion_requirements
                )
            if runtime_state.worktree is not None:
                if self.dependencies.worktree_manager is None:
                    raise ValueError("resume of a worktree session requires GitWorktreeManager")
                self.dependencies.worktree_manager.validate_session(runtime_state.worktree)
            interrupted = _repair_interrupted_tool_uses(snapshot.messages)
            if interrupted is not None:
                messages = [*snapshot.messages, interrupted, *params.messages]
                store.append_message(params.session_id, interrupted)
            for message in params.messages:
                store.append_message(params.session_id, message)
        else:
            assert isinstance(params, StartQueryParams)
            model = params.model
            system_prompt = params.system_prompt
            capabilities = params.capabilities
            completion_requirements = params.completion_requirements
            if store is not None:
                store.create(
                    SessionMetadata(
                        schema_version=4,
                        session_id=params.session_id,
                        repository_root=repository_root,
                        initial_working_directory=repository_root,
                        model=params.model,
                        system_prompt=params.system_prompt,
                        capabilities=params.capabilities,
                        completion_requirements=params.completion_requirements,
                    )
                )
                for message in messages:
                    store.append_message(params.session_id, message)
        transcript = SessionTranscript(session_id=params.session_id, messages=messages)
        working_directory = (
            runtime_state.worktree.path if runtime_state.worktree is not None else repository_root
        )
        runtime_state.instruction_state = self.dependencies.instruction_resolver.activate_for_path(
            Path(working_directory),
            ".",
            runtime_state.instruction_state,
        )
        tool_context = ToolUseContext(
            session_id=params.session_id,
            working_directory=working_directory,
            repository_root=repository_root,
            state_directory=self.dependencies.state_directory or repository_root,
            capabilities=capabilities,
            permission_mode=runtime_state.permission_mode,
            plan_path=runtime_state.plan_path,
            worktree=runtime_state.worktree,
            instruction_state=runtime_state.instruction_state,
            file_observations=runtime_state.file_observations,
            completion_requirements=completion_requirements,
            permission_grants=runtime_state.permission_grants,
        )
        if store is not None:
            store.save_state(
                params.session_id,
                _session_state(tool_context, status="running"),
            )
        started_at = self.dependencies.monotonic()
        force_compact_reason: str | None = None

        while True:
            failure = _budget_failure(state, params.config, started_at, self.dependencies.monotonic())
            if failure is not None:
                state.status = "failed"
                state.stop_reason = failure.reason
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="failed",
                            reason=failure.reason,
                        ),
                    )
                yield RunStopped(transition=failure)
                return

            state.round_count += 1
            projection = await self.dependencies.context_pipeline.project(
                transcript,
                config=params.config,
                working_directory=tool_context.working_directory,
                runtime_context=tool_context,
                force_reason=force_compact_reason,
            )
            state.input_tokens += projection.summary_input_tokens
            state.output_tokens += projection.summary_output_tokens
            force_compact_reason = None
            if projection.compacted:
                yield ContextCompacted(
                    before_messages=projection.source_message_count,
                    after_messages=len(projection.messages),
                    reason=projection.reason or "context_pipeline",
                )
            yield ModelRequestStarted(
                session_id=state.session_id,
                round_number=state.round_count,
            )

            request = ModelRequest(
                model=model,
                system_prompt=(
                    f"{system_prompt}\n\n{projection.system_reminder}"
                    if projection.system_reminder
                    else system_prompt
                ),
                messages=projection.messages,
                tools=self.dependencies.tool_registry.schemas(tool_context),
                max_output_tokens=params.config.max_output_tokens,
            )
            completed: ModelCompleted | None = None
            try:
                remaining = max(
                    params.config.deadline_seconds
                    - (self.dependencies.monotonic() - started_at),
                    0.001,
                )
                async with asyncio.timeout(remaining):
                    async for event in self.dependencies.model_gateway.stream(request):
                        if isinstance(event, ModelTextDelta):
                            yield AssistantDelta(text=event.text)
                        elif isinstance(event, ModelRetryScheduled):
                            yield event
                        else:
                            completed = event
            except TimeoutError:
                failure = Failed(
                    error_code="DEADLINE_EXCEEDED",
                    reason=f"query exceeded {params.config.deadline_seconds} seconds",
                    retryable=False,
                )
                state.status = "failed"
                state.stop_reason = failure.reason
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="failed",
                            reason=failure.reason,
                        ),
                    )
                yield RunStopped(transition=failure)
                return
            except asyncio.CancelledError:
                state.status = "cancelled"
                state.stop_reason = "model request was cancelled"
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="cancelled",
                            reason=state.stop_reason,
                        ),
                    )
                raise
            except ModelGatewayError as exc:
                if (
                    exc.code == "CONTEXT_LENGTH_EXCEEDED"
                    and state.reactive_compaction_count < params.config.max_reactive_compactions
                ):
                    state.reactive_compaction_count += 1
                    force_compact_reason = "reactive_compact"
                    continue
                failure = Failed(
                    error_code=exc.code,
                    reason=str(exc),
                    retryable=exc.retryable,
                )
                state.status = "failed"
                state.stop_reason = failure.reason
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="failed",
                            reason=failure.reason,
                        ),
                    )
                yield RunStopped(transition=failure)
                return
            except Exception as exc:  # noqa: BLE001 - 未知 Provider 错误在 Runtime 边界结构化。
                failure = Failed(
                    error_code="MODEL_REQUEST_FAILED",
                    reason=str(exc) or type(exc).__name__,
                    retryable=False,
                )
                state.status = "failed"
                state.stop_reason = failure.reason
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="failed",
                            reason=failure.reason,
                        ),
                    )
                yield RunStopped(transition=failure)
                return

            if completed is None:
                failure = Failed(
                    error_code="MODEL_STREAM_INCOMPLETE",
                    reason="model stream ended without a completed message",
                    retryable=True,
                )
                state.status = "failed"
                state.stop_reason = failure.reason
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="failed",
                            reason=failure.reason,
                        ),
                    )
                yield RunStopped(transition=failure)
                return

            transcript.append(completed.message)
            if store is not None:
                store.append_message(params.session_id, completed.message)
            yield AssistantMessageCompleted(message=completed.message)
            state.input_tokens += completed.input_tokens
            state.output_tokens += completed.output_tokens

            tool_calls = [
                block
                for block in completed.message.content
                if isinstance(block, ToolUseBlock)
            ]
            if not tool_calls:
                if completed.stop_reason == "max_tokens":
                    failure = Failed(
                        error_code="MODEL_MAX_TOKENS",
                        reason="model output reached max_tokens",
                        retryable=True,
                    )
                    state.status = "failed"
                    state.stop_reason = failure.reason
                    if store is not None:
                        store.save_state(
                            params.session_id,
                            _session_state(
                                tool_context,
                                status="failed",
                                reason=failure.reason,
                            ),
                        )
                    yield RunStopped(transition=failure)
                    return
                tool_context.transcript_messages = transcript.snapshot()
                stop_result = await self.dependencies.tool_executor.hooks.run_stop(
                    StopHookPayload(messages=transcript.snapshot()),
                    tool_context,
                )
                if stop_result.blocking_reasons:
                    reasons = tuple(stop_result.blocking_reasons)
                    state.stop_block_count = (
                        state.stop_block_count + 1
                        if reasons == state.last_stop_reasons
                        else 1
                    )
                    state.last_stop_reasons = reasons
                    blocking_message = RuntimeMessage(
                        role="user",
                        content=[
                            TextBlock(
                                text=(
                                    "<completion-gate>\n"
                                    + "\n".join(f"- {reason}" for reason in reasons)
                                    + "\nContinue working and satisfy these evidence requirements before finishing."
                                    "\n</completion-gate>"
                                )
                            )
                        ],
                    )
                    transcript.append(blocking_message)
                    if store is not None:
                        store.append_message(params.session_id, blocking_message)
                    if state.stop_block_count >= 3:
                        transition = Blocked(
                            reason="completion requirements remained unmet after three stop attempts"
                        )
                        state.status = "blocked"
                        state.stop_reason = transition.reason
                        if store is not None:
                            store.save_state(
                                params.session_id,
                                _session_state(
                                    tool_context,
                                    status="blocked",
                                    reason=transition.reason,
                                ),
                            )
                        yield RunStopped(transition=transition)
                        return
                    continue
                state.status = "completed"
                state.stop_reason = completed.stop_reason
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="completed",
                            reason=completed.stop_reason,
                        ),
                    )
                yield RunCompleted(transition=Complete(reason=completed.stop_reason))
                return

            signature = json.dumps(
                [{"name": call.name, "input": call.input} for call in tool_calls],
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature == state.last_tool_signature:
                state.no_progress_rounds += 1
            else:
                state.no_progress_rounds = 0
                state.last_tool_signature = signature

            for call in tool_calls:
                yield ToolRequested(call=call)

            tool_context.transcript_messages = transcript.snapshot()
            results = {}
            try:
                async for update in run_tools(
                    tool_calls,
                    registry=self.dependencies.tool_registry,
                    executor=self.dependencies.tool_executor,
                    context=tool_context,
                ):
                    tool_context = update.context
                    if store is not None:
                        store.save_state(
                            params.session_id,
                            SessionRuntimeState(
                                permission_mode=tool_context.permission_mode,
                                plan_path=tool_context.plan_path,
                                worktree=tool_context.worktree,
                                capabilities=tool_context.capabilities,
                                instruction_state=tool_context.instruction_state,
                                file_observations=tool_context.file_observations,
                                completion_requirements=tool_context.completion_requirements,
                                permission_grants=tool_context.permission_grants,
                                last_status="running",
                            ),
                        )
                    if update.result is not None and update.tool_use_id is not None:
                        results[update.tool_use_id] = update.result
                        yield ToolCompleted(
                            tool_use_id=update.tool_use_id,
                            result=update.result,
                        )
            except asyncio.CancelledError:
                for call in tool_calls:
                    results.setdefault(
                        call.id,
                        ToolResult(
                            error=ToolError(
                                code="TOOL_CANCELLED",
                                message="tool execution was cancelled",
                                retryable=True,
                            )
                        ),
                    )
                result_message = _tool_result_message(tool_calls, results)
                transcript.append(result_message)
                if store is not None:
                    store.append_message(params.session_id, result_message)
                state.status = "cancelled"
                state.stop_reason = "tool execution was cancelled"
                if store is not None:
                    store.save_state(
                        params.session_id,
                        _session_state(
                            tool_context,
                            status="cancelled",
                            reason=state.stop_reason,
                        ),
                    )
                yield RunStopped(transition=Cancelled(reason=state.stop_reason))
                return

            result_message = _tool_result_message(tool_calls, results)
            transcript.append(result_message)
            if store is not None:
                store.append_message(params.session_id, result_message)
            for call in tool_calls:
                for message in results[call.id].new_messages:
                    transcript.append(message)
                    if store is not None:
                        store.append_message(params.session_id, message)


def _tool_result_message(calls: list[ToolUseBlock], results: dict[str, ToolResult]) -> RuntimeMessage:
    return RuntimeMessage(
        role="user",
        content=[
            ToolResultBlock(
                tool_use_id=call.id,
                content=results[call.id].model_dump(mode="json"),
                is_error=results[call.id].error is not None,
            )
            for call in calls
        ],
    )


def _session_state(
    context: ToolUseContext,
    *,
    status: Literal["running", "completed", "blocked", "failed", "cancelled"],
    reason: str | None = None,
) -> SessionRuntimeState:
    return SessionRuntimeState(
        permission_mode=context.permission_mode,
        plan_path=context.plan_path,
        worktree=context.worktree,
        capabilities=context.capabilities,
        instruction_state=context.instruction_state,
        file_observations=context.file_observations,
        completion_requirements=context.completion_requirements,
        permission_grants=context.permission_grants,
        last_status=status,
        last_reason=reason,
    )


def _repair_interrupted_tool_uses(messages: list[RuntimeMessage]) -> RuntimeMessage | None:
    """只修复 transcript 尾部悬空调用；中间配对错误说明权威日志已损坏。"""

    known: set[str] = set()
    resolved: set[str] = set()
    pending: dict[str, ToolUseBlock] = {}
    for message in messages:
        if message.role == "assistant":
            if pending:
                raise ValueError("session contains unresolved tool calls before a later assistant message")
            for block in message.content:
                if not isinstance(block, ToolUseBlock):
                    continue
                if block.id in known:
                    raise ValueError(f"session contains duplicate tool use id: {block.id}")
                known.add(block.id)
                pending[block.id] = block
            continue
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            if block.tool_use_id not in known:
                raise ValueError(
                    f"session contains tool result without a matching tool use: {block.tool_use_id}"
                )
            if block.tool_use_id in resolved:
                raise ValueError(f"session contains duplicate tool result: {block.tool_use_id}")
            resolved.add(block.tool_use_id)
            pending.pop(block.tool_use_id, None)
    if not pending:
        return None
    interrupted = {
        tool_use_id: ToolResult(
            error=ToolError(
                code="PROCESS_INTERRUPTED",
                message="the previous process ended before this tool call completed",
                retryable=True,
            )
        )
        for tool_use_id in pending
    }
    return _tool_result_message(list(pending.values()), interrupted)


def _budget_failure(
    state: QueryState,
    config: QueryConfig,
    started_at: float,
    now: float,
) -> Failed | None:
    if state.round_count >= config.max_rounds:
        return Failed(
            error_code="MAX_ROUNDS",
            reason=f"query exceeded {config.max_rounds} rounds",
        )
    if state.input_tokens + state.output_tokens >= config.max_total_tokens:
        return Failed(
            error_code="MAX_TOTAL_TOKENS",
            reason=f"query exceeded {config.max_total_tokens} tokens",
        )
    if now - started_at >= config.deadline_seconds:
        return Failed(
            error_code="DEADLINE_EXCEEDED",
            reason=f"query exceeded {config.deadline_seconds} seconds",
        )
    if state.no_progress_rounds >= config.max_no_progress_rounds:
        return Failed(
            error_code="NO_PROGRESS",
            reason=f"query repeated the same tool calls for {state.no_progress_rounds} rounds",
        )
    return None
