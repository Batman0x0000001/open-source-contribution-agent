"""实现模型请求、工具执行、结果回填和停止判定的主循环。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

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
from osc_agent.runtime.session import SessionMetadata
from osc_agent.runtime.session_store import SessionStore
from osc_agent.runtime.state import (
    AgentRunState,
    InstructionsActivated,
    PlanModeEntered,
    PlanSaved,
)
from osc_agent.runtime.tool_models import ToolError, ToolResult
from osc_agent.runtime.tool_execution import PostToolUseHookError
from osc_agent.runtime.tool_orchestration import (
    ToolBatchStateCommitted,
    ToolResultAvailable,
    run_tools,
)


@dataclass
class _QueryProgress:
    session_id: str
    round_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reactive_compaction_count: int = 0
    no_progress_rounds: int = 0
    last_tool_signature: str | None = None
    stop_block_count: int = 0
    last_stop_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _OpenedSession:
    messages: list[RuntimeMessage]
    model: str
    system_prompt: str
    agent_state: AgentRunState


@dataclass
class _ActiveQuery:
    params: QueryParams
    progress: _QueryProgress
    transcript: SessionTranscript
    agent_state: AgentRunState
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
        self.agent_state = self.agent_state.with_status(status, reason)
        if self.store is not None:
            self.store.save_state(self.params.session_id, self.agent_state)


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
                        snapshot.state.with_status("cancelled", "query was cancelled"),
                    )
                raise

    async def _query_locked(self, params: QueryParams) -> AsyncIterator[RuntimeEvent]:
        run = self._open_run(params)
        progress = run.progress
        transcript = run.transcript
        model = run.model
        system_prompt = run.system_prompt
        started_at = run.started_at
        force_compact_reason: str | None = None

        while True:
            tool_context = run.agent_state.tool_context(
                session_id=params.session_id,
                state_directory=self.dependencies.state_directory,
                transcript_messages=transcript.snapshot(),
            )
            failure = _budget_failure(
                progress,
                params.config,
                started_at,
                self.dependencies.monotonic(),
            )
            if failure is not None:
                yield _stopped(run, failure)
                return

            progress.round_count += 1
            try:
                projection = await self.dependencies.context_pipeline.project(
                    transcript,
                    config=params.config,
                    runtime_context=tool_context,
                    instruction_resolver=self.dependencies.instruction_resolver,
                    force_reason=force_compact_reason,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - Context 边界异常必须形成终态。
                yield _stopped(
                    run,
                    Failed(
                        error_code="CONTEXT_PROJECTION_FAILED",
                        reason=str(exc) or type(exc).__name__,
                        retryable=False,
                    ),
                )
                return
            progress.input_tokens += projection.summary_input_tokens
            progress.output_tokens += projection.summary_output_tokens
            force_compact_reason = None
            if projection.compacted:
                yield ContextCompacted(
                    before_messages=projection.source_message_count,
                    after_messages=len(projection.messages),
                    reason=projection.reason or "context_pipeline",
                )
            yield ModelRequestStarted(
                session_id=progress.session_id,
                round_number=progress.round_count,
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
                        elif isinstance(event, ModelCompleted):
                            completed = event
                        else:
                            raise TypeError(
                                f"unsupported model event: {type(event).__name__}"
                            )
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
                    and progress.reactive_compaction_count
                    < params.config.max_reactive_compactions
                ):
                    progress.reactive_compaction_count += 1
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
            yield AssistantMessageCompleted(
                message=completed.message.model_copy(deep=True)
            )
            progress.input_tokens += completed.input_tokens
            progress.output_tokens += completed.output_tokens

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
                try:
                    stop_result = await self.dependencies.tool_executor.hooks.run_stop(
                        StopHookPayload(messages=transcript.snapshot()),
                        tool_context,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - 完成门禁异常必须失败关闭。
                    yield _stopped(
                        run,
                        Failed(
                            error_code="STOP_HOOK_FAILED",
                            reason=str(exc) or type(exc).__name__,
                            retryable=False,
                        ),
                    )
                    return
                if stop_result.blocking_reasons:
                    reasons = tuple(stop_result.blocking_reasons)
                    progress.stop_block_count = (
                        progress.stop_block_count + 1
                        if reasons == progress.last_stop_reasons
                        else 1
                    )
                    progress.last_stop_reasons = reasons
                    run.append(_completion_gate_message(reasons))
                    if progress.stop_block_count >= 3:
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
            if signature == progress.last_tool_signature:
                progress.no_progress_rounds += 1
            else:
                progress.no_progress_rounds = 0
                progress.last_tool_signature = signature

            for call in tool_calls:
                yield ToolRequested(call=call.model_copy(deep=True))

            results: dict[str, ToolResult] = {}
            try:
                async for update in run_tools(
                    tool_calls,
                    executor=self.dependencies.tool_executor,
                    state=run.agent_state,
                    session_id=params.session_id,
                    state_directory=self.dependencies.state_directory,
                    transcript_messages=transcript.snapshot(),
                ):
                    if isinstance(update, ToolBatchStateCommitted):
                        run.agent_state = update.agent_state
                        run.persist("running")
                    elif isinstance(update, ToolResultAvailable):
                        results[update.tool_use_id] = update.result
                        yield ToolCompleted(
                            tool_use_id=update.tool_use_id,
                            result=update.result.model_copy(deep=True),
                        )
                    else:
                        raise TypeError(
                            f"unsupported tool execution event: {type(update).__name__}"
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
                _append_tool_results(run, tool_calls, results)
                yield _stopped(run, Cancelled(reason="tool execution was cancelled"))
                return
            except PostToolUseHookError as exc:
                for call in tool_calls:
                    results.setdefault(
                        call.id,
                        ToolResult(
                            error=ToolError(
                                code="TOOL_NOT_EXECUTED",
                                message=(
                                    "tool was not executed because a PostToolUse Hook "
                                    f"failed after {exc.tool_use_id}"
                                ),
                                retryable=False,
                            )
                        ),
                    )
                _append_tool_results(run, tool_calls, results)
                yield _stopped(
                    run,
                    Failed(
                        error_code="POST_TOOL_HOOK_FAILED",
                        reason=(
                            f"PostToolUse Hook failed after {exc.tool_use_id}: "
                            f"{str(exc) or type(exc).__name__}"
                        ),
                        retryable=False,
                    ),
                )
                return

            _append_tool_results(run, tool_calls, results)

    def _open_run(self, params: QueryParams) -> _ActiveQuery:
        store = self.dependencies.session_store
        workspace_root = str(Path(params.workspace_root).resolve())
        if isinstance(params, ResumeQueryParams):
            opened = self._resume(params, workspace_root)
        else:
            assert isinstance(params, StartQueryParams)
            opened = self._start(params, workspace_root)
        agent_state = opened.agent_state
        working_directory = agent_state.workspace.working_directory
        instruction_state = self.dependencies.instruction_resolver.activate_for_path(
            Path(working_directory),
            ".",
            agent_state.workspace.instruction_state,
        )
        agent_state = agent_state.apply(
            InstructionsActivated(state=instruction_state, replace=True)
        )
        run = _ActiveQuery(
            params=params,
            progress=_QueryProgress(session_id=params.session_id),
            transcript=SessionTranscript(session_id=params.session_id, messages=opened.messages),
            agent_state=agent_state,
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
        instruction_state = self.dependencies.instruction_resolver.activate_root(
            Path(workspace_root)
        )
        agent_state = AgentRunState.start(
            workspace_root=workspace_root,
            capabilities=params.capabilities,
            completion_requirements=params.completion_requirements,
            instruction_state=instruction_state,
        )
        if params.start_in_plan_mode:
            plan_name = _plan_name(params.session_id)
            plan = Path(self.dependencies.state_directory) / "plans" / plan_name
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.touch(exist_ok=True)
            agent_state = agent_state.apply_all(
                (PlanModeEntered(), PlanSaved(path=plan_name))
            )
        if store is not None:
            store.create(
                SessionMetadata(
                    schema_version=6,
                    session_id=params.session_id,
                    workspace_root=workspace_root,
                    model=params.model,
                    system_prompt=params.system_prompt,
                ),
                agent_state,
            )
            for message in messages:
                store.append_message(params.session_id, message)
        return _OpenedSession(
            messages=messages,
            model=params.model,
            system_prompt=params.system_prompt,
            agent_state=agent_state,
        )

    def _resume(self, params: ResumeQueryParams, workspace_root: str) -> _OpenedSession:
        store = self.dependencies.session_store
        if store is None:
            raise ValueError("resume requires a SessionStore")
        snapshot = store.load(params.session_id)
        if snapshot is None:
            raise ValueError(f"unknown session: {params.session_id}")
        if str(Path(snapshot.metadata.workspace_root).resolve()) != workspace_root:
            raise ValueError("resume workspace does not match the saved session")

        agent_state = snapshot.state
        if agent_state.workspace.worktree is not None:
            if self.dependencies.workspace_validator is None:
                raise ValueError("resume of a worktree session requires a workspace validator")
            self.dependencies.workspace_validator.validate_session(
                agent_state.workspace.worktree
            )

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
            agent_state=agent_state,
        )


def _plan_name(session_id: str) -> str:
    if not session_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in session_id
    ):
        raise ValueError("plan session id contains unsafe path characters")
    return f"{session_id}.md"


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


def _append_tool_results(
    run: _ActiveQuery,
    calls: list[ToolUseBlock],
    results: dict[str, ToolResult],
) -> None:
    run.append(_tool_result_message(calls, results))
    for call in calls:
        for message in results[call.id].new_messages:
            run.append(message)


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
    progress: _QueryProgress,
    config: QueryConfig,
    started_at: float,
    now: float,
) -> Failed | None:
    if progress.round_count >= config.max_rounds:
        return Failed(
            error_code="MAX_ROUNDS",
            reason=f"query exceeded {config.max_rounds} rounds",
        )
    if progress.input_tokens + progress.output_tokens >= config.max_total_tokens:
        return Failed(
            error_code="MAX_TOTAL_TOKENS",
            reason=f"query exceeded {config.max_total_tokens} tokens",
        )
    if now - started_at >= config.deadline_seconds:
        return Failed(
            error_code="DEADLINE_EXCEEDED",
            reason=f"query exceeded {config.deadline_seconds} seconds",
        )
    if progress.no_progress_rounds >= config.max_no_progress_rounds:
        return Failed(
            error_code="NO_PROGRESS",
            reason=(
                "query repeated the same tool calls for "
                f"{progress.no_progress_rounds} rounds"
            ),
        )
    return None
