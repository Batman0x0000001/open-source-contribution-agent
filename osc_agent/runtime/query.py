from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from osc_agent.runtime.context import SessionTranscript
from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelCompleted, ModelGatewayError, ModelRequest, ModelTextDelta
from osc_agent.runtime.models import (
    AssistantDelta,
    AssistantMessageCompleted,
    Cancelled,
    Complete,
    ContextCompacted,
    Failed,
    ModelRequestStarted,
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
from osc_agent.runtime.tool_orchestration import run_tools


class AgentRuntime:
    def __init__(self, dependencies: QueryDependencies) -> None:
        self.dependencies = dependencies

    async def query(self, params: QueryParams) -> AsyncIterator[RuntimeEvent]:
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
            runtime_state = snapshot.runtime_state
            if runtime_state.capabilities is not None:
                capabilities = capabilities.intersect(runtime_state.capabilities)
            if runtime_state.worktree is not None:
                if self.dependencies.worktree_manager is None:
                    raise ValueError("resume of a worktree session requires WorktreeManager")
                self.dependencies.worktree_manager.validate_session(runtime_state.worktree)
            for message in params.messages:
                store.append_message(params.session_id, message)
        else:
            assert isinstance(params, StartQueryParams)
            model = params.model
            system_prompt = params.system_prompt
            capabilities = params.capabilities
            if store is not None:
                store.create(
                    SessionMetadata(
                        session_id=params.session_id,
                        repository_root=repository_root,
                        initial_working_directory=repository_root,
                        model=params.model,
                        system_prompt=params.system_prompt,
                        capabilities=params.capabilities,
                    )
                )
                for message in messages:
                    store.append_message(params.session_id, message)
        transcript = SessionTranscript(session_id=params.session_id, messages=messages)
        working_directory = (
            runtime_state.worktree.path if runtime_state.worktree is not None else repository_root
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
        )
        started_at = self.dependencies.monotonic()
        force_compact_reason: str | None = None

        while True:
            failure = _budget_failure(state, params.config, started_at, self.dependencies.monotonic())
            if failure is not None:
                state.status = "failed"
                state.stop_reason = failure.reason
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
                async for event in self.dependencies.model_gateway.stream(request):
                    if isinstance(event, ModelTextDelta):
                        yield AssistantDelta(text=event.text)
                    else:
                        completed = event
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
                    yield RunStopped(transition=failure)
                    return
                state.status = "completed"
                state.stop_reason = completed.stop_reason
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
