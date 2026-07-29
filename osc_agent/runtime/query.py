"""实现模型请求、工具执行、结果回填和停止判定的主循环。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

from osc_agent.completion.models import CompletionRequirements
from osc_agent.runtime.context import SessionTranscript
from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelCompleted, ModelGatewayError, ModelRequest, ModelTextDelta
from osc_agent.runtime.events import (
    AssistantDelta,
    AssistantMessageCompleted,
    Blocked,
    Cancelled,
    Complete,
    ContextCompacted,
    Failed,
    ModelRequestStarted,
    ModelRetryScheduled,
    RunCompleted,
    RunStopped,
    RuntimeEvent,
    ToolCompleted,
    ToolRequested,
)
from osc_agent.runtime.hooks import StopHookPayload
from osc_agent.runtime.messages import (
    RuntimeMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from osc_agent.runtime.query_models import (
    QueryConfig,
    QueryParams,
    ResumeQueryParams,
    StartQueryParams,
)
from osc_agent.runtime.session import SessionMetadata, SessionRuntimeState
from osc_agent.runtime.session_store import SessionStore
from osc_agent.runtime.tool_models import CapabilityScope, ToolError, ToolResult, ToolUseContext
from osc_agent.runtime.tool_orchestration import run_tools


@dataclass
class _QueryState:
    session_id: str
    round_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reactive_compaction_count: int = 0
    no_progress_rounds: int = 0
    last_tool_signature: str | None = None
    stop_block_count: int = 0
    last_stop_reasons: tuple[str, ...] = ()
    status: Literal["running", "completed", "blocked", "failed", "cancelled"] = "running"
    stop_reason: str | None = None


@dataclass(frozen=True)
class _OpenedSession:
    messages: list[RuntimeMessage]
    model: str
    system_prompt: str
    runtime_state: SessionRuntimeState
    capabilities: CapabilityScope
    completion_requirements: CompletionRequirements


@dataclass
class _ActiveQuery:
    params: QueryParams
    state: _QueryState
    transcript: SessionTranscript
    tool_context: ToolUseContext
    model: str
    system_prompt: str
    store: SessionStore | None
    started_at: float

    def append(self, message: RuntimeMessage) -> None:
        self.transcript.append(message)
        if self.store is not None:
            self.store.append_message(self.params.session_id, message)

    def persist(
        self,
        status: Literal["running", "completed", "blocked", "failed", "cancelled"],
        reason: str | None = None,
    ) -> None:
        self.state.status = status
        self.state.stop_reason = reason
        if self.store is not None:
            self.store.save_state(
                self.params.session_id,
                _session_state(self.tool_context, status=status, reason=reason),
            )


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
        run = self._open_run(params)
        state = run.state
        transcript = run.transcript
        tool_context = run.tool_context
        model = run.model
        system_prompt = run.system_prompt
        started_at = run.started_at
        force_compact_reason: str | None = None

        while True:
            failure = _budget_failure(state, params.config, started_at, self.dependencies.monotonic())
            if failure is not None:
                yield _stopped(run, failure)
                return

            state.round_count += 1
            projection = await self.dependencies.context_pipeline.project(
                transcript,
                config=params.config,
                working_directory=tool_context.working_directory,
                runtime_context=tool_context,
                instruction_resolver=self.dependencies.instruction_resolver,
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
                tools=self.dependencies.tool_executor.registry.schemas(tool_context),
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
                yield _stopped(
                    run,
                    Failed(
                        error_code="DEADLINE_EXCEEDED",
                        reason=f"query exceeded {params.config.deadline_seconds} seconds",
                        retryable=False,
                    ),
                )
                return
            except asyncio.CancelledError:
                raise
            except ModelGatewayError as exc:
                if (
                    exc.code == "CONTEXT_LENGTH_EXCEEDED"
                    and state.reactive_compaction_count < params.config.max_reactive_compactions
                ):
                    state.reactive_compaction_count += 1
                    force_compact_reason = "reactive_compact"
                    continue
                yield _stopped(
                    run,
                    Failed(error_code=exc.code, reason=str(exc), retryable=exc.retryable),
                )
                return
            except Exception as exc:  # noqa: BLE001 - 未知 Provider 错误在 Runtime 边界结构化。
                yield _stopped(
                    run,
                    Failed(
                        error_code="MODEL_REQUEST_FAILED",
                        reason=str(exc) or type(exc).__name__,
                        retryable=False,
                    ),
                )
                return

            if completed is None:
                yield _stopped(
                    run,
                    Failed(
                        error_code="MODEL_STREAM_INCOMPLETE",
                        reason="model stream ended without a completed message",
                        retryable=True,
                    ),
                )
                return

            run.append(completed.message)
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
                    yield _stopped(
                        run,
                        Failed(
                            error_code="MODEL_MAX_TOKENS",
                            reason="model output reached max_tokens",
                            retryable=True,
                        ),
                    )
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
                    run.append(_completion_gate_message(reasons))
                    if state.stop_block_count >= 3:
                        yield _stopped(
                            run,
                            Blocked(
                                reason="completion requirements remained unmet after three stop attempts"
                            ),
                        )
                        return
                    continue
                yield _completed(run, completed.stop_reason)
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
            results: dict[str, ToolResult] = {}
            try:
                async for update in run_tools(
                    tool_calls,
                    executor=self.dependencies.tool_executor,
                    context=tool_context,
                ):
                    tool_context = update.context
                    run.tool_context = tool_context
                    run.persist("running")
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
                run.append(_tool_result_message(tool_calls, results))
                yield _stopped(run, Cancelled(reason="tool execution was cancelled"))
                return

            run.append(_tool_result_message(tool_calls, results))
            for call in tool_calls:
                for message in results[call.id].new_messages:
                    run.append(message)

    def _open_run(self, params: QueryParams) -> _ActiveQuery:
        store = self.dependencies.session_store
        workspace_root = str(Path(params.workspace_root).resolve())
        if isinstance(params, ResumeQueryParams):
            opened = self._resume(params, workspace_root)
        else:
            assert isinstance(params, StartQueryParams)
            opened = self._start(params, workspace_root)
        runtime_state = opened.runtime_state
        working_directory = (
            runtime_state.worktree.path if runtime_state.worktree is not None else workspace_root
        )
        runtime_state.instruction_state = self.dependencies.instruction_resolver.activate_for_path(
            Path(working_directory),
            ".",
            runtime_state.instruction_state,
        )
        tool_context = ToolUseContext(
            session_id=params.session_id,
            working_directory=working_directory,
            state_directory=self.dependencies.state_directory,
            capabilities=opened.capabilities,
            permission_mode=runtime_state.permission_mode,
            plan_path=runtime_state.plan_path,
            worktree=runtime_state.worktree,
            instruction_state=runtime_state.instruction_state,
            file_observations=runtime_state.file_observations,
            completion_requirements=opened.completion_requirements,
            permission_grants=runtime_state.permission_grants,
        )
        run = _ActiveQuery(
            params=params,
            state=_QueryState(session_id=params.session_id),
            transcript=SessionTranscript(session_id=params.session_id, messages=opened.messages),
            tool_context=tool_context,
            model=opened.model,
            system_prompt=opened.system_prompt,
            store=store,
            started_at=self.dependencies.monotonic(),
        )
        run.persist("running")
        return run

    def _start(self, params: StartQueryParams, workspace_root: str) -> _OpenedSession:
        store = self.dependencies.session_store
        messages = list(params.messages)
        if store is not None:
            store.create(
                SessionMetadata(
                    schema_version=4,
                    session_id=params.session_id,
                    repository_root=workspace_root,
                    initial_working_directory=workspace_root,
                    model=params.model,
                    system_prompt=params.system_prompt,
                    capabilities=params.capabilities,
                    completion_requirements=params.completion_requirements,
                )
            )
            for message in messages:
                store.append_message(params.session_id, message)
        return _OpenedSession(
            messages=messages,
            model=params.model,
            system_prompt=params.system_prompt,
            runtime_state=SessionRuntimeState(),
            capabilities=params.capabilities,
            completion_requirements=params.completion_requirements,
        )

    def _resume(self, params: ResumeQueryParams, workspace_root: str) -> _OpenedSession:
        store = self.dependencies.session_store
        if store is None:
            raise ValueError("resume requires a SessionStore")
        snapshot = store.load(params.session_id)
        if snapshot is None:
            raise ValueError(f"unknown session: {params.session_id}")
        if str(Path(snapshot.metadata.repository_root).resolve()) != workspace_root:
            raise ValueError("resume workspace does not match the saved session")

        runtime_state = snapshot.runtime_state
        capabilities = snapshot.metadata.capabilities
        requirements = snapshot.metadata.completion_requirements
        if runtime_state.capabilities is not None:
            capabilities = capabilities.intersect(runtime_state.capabilities)
        if runtime_state.completion_requirements is not None:
            requirements = requirements.tighten(runtime_state.completion_requirements)
        if runtime_state.worktree is not None:
            if self.dependencies.workspace_validator is None:
                raise ValueError("resume of a worktree session requires a workspace validator")
            self.dependencies.workspace_validator.validate_session(runtime_state.worktree)

        messages = [*snapshot.messages, *params.messages]
        interrupted = _repair_interrupted_tool_uses(snapshot.messages)
        if interrupted is not None:
            messages = [*snapshot.messages, interrupted, *params.messages]
            store.append_message(params.session_id, interrupted)
        for message in params.messages:
            store.append_message(params.session_id, message)
        return _OpenedSession(
            messages=messages,
            model=snapshot.metadata.model,
            system_prompt=snapshot.metadata.system_prompt,
            runtime_state=runtime_state,
            capabilities=capabilities,
            completion_requirements=requirements,
        )


def _stopped(
    run: _ActiveQuery,
    transition: Blocked | Failed | Cancelled,
) -> RunStopped:
    status: Literal["blocked", "failed", "cancelled"]
    if isinstance(transition, Blocked):
        status = "blocked"
    elif isinstance(transition, Failed):
        status = "failed"
    else:
        status = "cancelled"
    run.persist(status, transition.reason)
    return RunStopped(transition=transition)


def _completed(run: _ActiveQuery, reason: str) -> RunCompleted:
    run.persist("completed", reason)
    return RunCompleted(transition=Complete(reason=reason))


def _completion_gate_message(reasons: tuple[str, ...]) -> RuntimeMessage:
    return RuntimeMessage(
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
    state: _QueryState,
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
