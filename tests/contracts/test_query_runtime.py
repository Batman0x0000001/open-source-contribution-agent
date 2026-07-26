from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import AsyncIterator

import pytest

from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import (
    ModelCompleted,
    ModelEvent,
    ModelGatewayError,
    ModelRequest,
    ModelTextDelta,
)
from osc_agent.runtime.models import (
    CapabilityScope,
    CompletionRequirements,
    ContractModel,
    ContextUpdate,
    QueryConfig,
    QueryParams,
    ResumeQueryParams,
    StartQueryParams,
    RunCompleted,
    RunStopped,
    RuntimeMessage,
    SessionMetadata,
    TextBlock,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
)
from osc_agent.runtime.completion import CompletionEvidenceStopHook
from osc_agent.runtime.hooks import HookRegistry
from osc_agent.runtime.query import AgentRuntime
from osc_agent.runtime.tool import BaseTool, ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.runtime.session_store import FileSessionStore


class EchoInput(ContractModel):
    value: str


class EchoOutput(ContractModel):
    value: str


class EchoTool(BaseTool[EchoInput, EchoOutput]):
    name = "echo"
    input_model = EchoInput
    output_model = EchoOutput

    async def call(self, input: EchoInput, context: ToolUseContext) -> ToolResult:
        return ToolResult(data={"value": input.value})


class NarrowTool(EchoTool):
    name = "narrow"

    async def call(self, input: EchoInput, context: ToolUseContext) -> ToolResult:
        return ToolResult(
            data={"value": input.value},
            context_update=ContextUpdate(
                capabilities=CapabilityScope(allowed_tools=frozenset({"echo"}))
            ),
        )


class FakeGateway:
    def __init__(self, turns: list[list[ModelEvent]]) -> None:
        self.turns = list(turns)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        for event in self.turns.pop(0):
            yield event


def params(*, max_rounds: int = 30) -> QueryParams:
    return StartQueryParams(
        session_id="session-1",
        model="test-model",
        messages=[RuntimeMessage(role="user", content=[TextBlock(text="hello")])],
        repository_root="C:/repo",
        config=QueryConfig(max_rounds=max_rounds),
    )


async def collect(runtime: AgentRuntime, query_params: QueryParams):
    return [event async for event in runtime.query(query_params)]


def runtime(gateway: FakeGateway, tools=None, session_store=None, hooks=None) -> AgentRuntime:
    registry = ToolRegistry(tools or [])
    return AgentRuntime(
        QueryDependencies(
            model_gateway=gateway,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry, hooks=hooks),
            session_store=session_store,
        )
    )


def test_query_is_an_async_generator() -> None:
    assert inspect.isasyncgenfunction(AgentRuntime.query)


def test_query_streams_text_and_completes() -> None:
    gateway = FakeGateway(
        [[
            ModelTextDelta(text="done"),
            ModelCompleted(
                message=RuntimeMessage(role="assistant", content=[TextBlock(text="done")]),
                stop_reason="end_turn",
                input_tokens=3,
                output_tokens=1,
            ),
        ]]
    )

    events = asyncio.run(collect(runtime(gateway), params()))

    assert [event.type for event in events] == [
        "model_request_started",
        "assistant_delta",
        "assistant_message_completed",
        "run_completed",
    ]
    assert isinstance(events[-1], RunCompleted)


def test_completion_gate_blocks_three_identical_stop_attempts() -> None:
    final = ModelCompleted(
        message=RuntimeMessage(
            role="assistant",
            content=[TextBlock(text="done without evidence")],
        ),
        stop_reason="end_turn",
    )
    gateway = FakeGateway([[final], [final], [final]])
    hooks = HookRegistry()
    hooks.register_stop(CompletionEvidenceStopHook())
    query_params = StartQueryParams(
        session_id="completion-gate",
        model="test-model",
        messages=[RuntimeMessage(role="user", content=[TextBlock(text="implement")])],
        repository_root="C:/repo",
        completion_requirements=CompletionRequirements(
            required_evidence=frozenset(
                {"successful_test", "git_change_snapshot"}
            )
        ),
    )

    events = asyncio.run(
        collect(runtime(gateway, hooks=hooks), query_params)
    )

    assert isinstance(events[-1], RunStopped)
    assert events[-1].transition.kind == "blocked"
    assert len(gateway.requests) == 3


def test_tool_result_is_appended_in_protocol_order_before_next_round() -> None:
    call = ToolUseBlock(id="call-1", name="echo", input={"value": "ok"})
    gateway = FakeGateway(
        [
            [
                ModelCompleted(
                    message=RuntimeMessage(role="assistant", content=[call]),
                    stop_reason="tool_use",
                )
            ],
            [
                ModelCompleted(
                    message=RuntimeMessage(role="assistant", content=[TextBlock(text="finished")]),
                    stop_reason="end_turn",
                )
            ],
        ]
    )

    events = asyncio.run(collect(runtime(gateway, [EchoTool()]), params()))

    second_request = gateway.requests[1]
    result_message = second_request.messages[-1]
    assert result_message.role == "user"
    assert isinstance(result_message.content[0], ToolResultBlock)
    assert result_message.content[0].tool_use_id == "call-1"
    assert result_message.content[0].is_error is False
    assert [event.type for event in events] == [
        "model_request_started",
        "assistant_message_completed",
        "tool_requested",
        "tool_completed",
        "model_request_started",
        "assistant_message_completed",
        "run_completed",
    ]


def test_model_exception_becomes_structured_terminal_event() -> None:
    class FailingGateway:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            raise RuntimeError("provider failed")
            yield

    registry = ToolRegistry()
    agent_runtime = AgentRuntime(
        QueryDependencies(
            model_gateway=FailingGateway(),
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
        )
    )

    events = asyncio.run(collect(agent_runtime, params()))

    assert isinstance(events[-1], RunStopped)
    assert events[-1].transition.kind == "failed"
    assert events[-1].transition.error_code == "MODEL_REQUEST_FAILED"


def test_round_budget_stops_before_an_unbounded_loop() -> None:
    call = ToolUseBlock(id="call-1", name="echo", input={"value": "ok"})
    gateway = FakeGateway(
        [[ModelCompleted(message=RuntimeMessage(role="assistant", content=[call]), stop_reason="tool_use")]]
    )

    events = asyncio.run(collect(runtime(gateway, [EchoTool()]), params(max_rounds=1)))

    assert isinstance(events[-1], RunStopped)
    assert events[-1].transition.error_code == "MAX_ROUNDS"


def test_context_length_error_reactively_compacts_once_then_recovers() -> None:
    class RecoveringGateway:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelGatewayError(
                    code="CONTEXT_LENGTH_EXCEEDED",
                    message="prompt is too long",
                    retryable=True,
                )
            yield ModelCompleted(
                message=RuntimeMessage(role="assistant", content=[TextBlock(text="recovered")]),
                stop_reason="end_turn",
            )

    gateway = RecoveringGateway()
    events = asyncio.run(collect(runtime(gateway), params()))

    assert [event.type for event in events] == [
        "model_request_started",
        "context_compacted",
        "model_request_started",
        "assistant_message_completed",
        "run_completed",
    ]
    assert len(gateway.requests[1].messages) == 1
    assert "<session_runtime>" in gateway.requests[1].system_prompt


def test_reactive_compact_has_a_strict_attempt_limit() -> None:
    class OversizedGateway:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            raise ModelGatewayError(
                code="CONTEXT_LENGTH_EXCEEDED",
                message="still too long",
                retryable=True,
            )
            yield

    events = asyncio.run(collect(runtime(OversizedGateway()), params()))

    assert isinstance(events[-1], RunStopped)
    assert events[-1].transition.error_code == "CONTEXT_LENGTH_EXCEEDED"
    assert [event.type for event in events].count("context_compacted") == 1


def test_repeated_identical_tool_calls_stop_as_no_progress() -> None:
    turns = []
    for index in range(4):
        call = ToolUseBlock(id=f"call-{index}", name="echo", input={"value": "same"})
        turns.append([ModelCompleted(message=RuntimeMessage(role="assistant", content=[call]), stop_reason="tool_use")])
    gateway = FakeGateway(turns)
    query_params = params()

    events = asyncio.run(collect(runtime(gateway, [EchoTool()]), query_params))

    assert isinstance(events[-1], RunStopped)
    assert events[-1].transition.error_code == "NO_PROGRESS"


def test_resume_replays_complete_tool_pairs_and_appends_new_user_message(tmp_path: Path) -> None:
    call = ToolUseBlock(id="call-1", name="echo", input={"value": "saved"})
    store = FileSessionStore(tmp_path)
    first_gateway = FakeGateway([
        [ModelCompleted(message=RuntimeMessage(role="assistant", content=[call]), stop_reason="tool_use")],
        [ModelCompleted(message=RuntimeMessage(role="assistant", content=[TextBlock(text="first done")]), stop_reason="end_turn")],
    ])
    initial = StartQueryParams(
        session_id="resume-1",
        model="saved-model",
        system_prompt="saved system prompt",
        capabilities=CapabilityScope(allowed_tools=frozenset({"echo"})),
        messages=[RuntimeMessage(role="user", content=[TextBlock(text="initial")])],
        repository_root=str(tmp_path),
    )
    asyncio.run(collect(runtime(first_gateway, [EchoTool()], store), initial))

    resumed_gateway = FakeGateway([
        [ModelCompleted(message=RuntimeMessage(role="assistant", content=[TextBlock(text="resumed")]), stop_reason="end_turn")]
    ])
    resumed = ResumeQueryParams(
        session_id="resume-1",
        messages=[RuntimeMessage(role="user", content=[TextBlock(text="continue")])],
        repository_root=str(tmp_path),
    )
    asyncio.run(collect(runtime(resumed_gateway, [EchoTool()], store), resumed))

    request = resumed_gateway.requests[0]
    assert request.model == "saved-model"
    assert request.system_prompt.startswith("saved system prompt")
    assert "<session_runtime>" in request.system_prompt
    assert [tool["name"] for tool in request.tools] == ["echo"]
    assert any(isinstance(block, ToolUseBlock) and block.id == "call-1" for message in request.messages for block in message.content)
    assert any(isinstance(block, ToolResultBlock) and block.tool_use_id == "call-1" for message in request.messages for block in message.content)
    assert request.messages[-1].content[0].text == "continue"


def test_resume_repairs_interrupted_tool_use_before_new_user_message(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path / "sessions")
    store.create(
        SessionMetadata(
            schema_version=4,
            session_id="interrupted",
            repository_root=str(tmp_path),
            initial_working_directory=str(tmp_path),
            model="saved-model",
            capabilities=CapabilityScope(allowed_tools=frozenset({"echo"})),
        )
    )
    store.append_message(
        "interrupted",
        RuntimeMessage(role="user", content=[TextBlock(text="initial")]),
    )
    store.append_message(
        "interrupted",
        RuntimeMessage(
            role="assistant",
            content=[ToolUseBlock(id="dangling", name="echo", input={"value": "x"})],
        ),
    )
    gateway = FakeGateway(
        [[ModelCompleted(message=RuntimeMessage(role="assistant", content=[TextBlock(text="done")]), stop_reason="end_turn")]]
    )

    asyncio.run(
        collect(
            runtime(gateway, [EchoTool()], store),
            ResumeQueryParams(
                session_id="interrupted",
                repository_root=str(tmp_path),
                messages=[RuntimeMessage(role="user", content=[TextBlock(text="continue")])],
            ),
        )
    )

    request = gateway.requests[0]
    repair = next(
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock) and block.tool_use_id == "dangling"
    )
    assert repair.is_error is True
    assert repair.content["error"]["code"] == "PROCESS_INTERRUPTED"
    assert request.messages[-1].content[0].text == "continue"


def test_resume_rejects_a_different_repository(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path / "sessions")
    start_gateway = FakeGateway([
        [ModelCompleted(message=RuntimeMessage(role="assistant", content=[TextBlock(text="done")]), stop_reason="end_turn")]
    ])
    asyncio.run(
        collect(
            runtime(start_gateway, session_store=store),
            StartQueryParams(
                session_id="repo-bound",
                model="saved",
                messages=[RuntimeMessage(role="user", content=[TextBlock(text="start")])],
                repository_root=str(tmp_path / "repo-a"),
            ),
        )
    )
    resume_gateway = FakeGateway([])
    try:
        asyncio.run(
            collect(
                runtime(resume_gateway, session_store=store),
                ResumeQueryParams(session_id="repo-bound", repository_root=str(tmp_path / "repo-b")),
            )
        )
    except ValueError as exc:
        assert "repository does not match" in str(exc)
    else:
        raise AssertionError("cross-repository resume should fail")


def test_resume_keeps_runtime_capability_narrowing(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path / "sessions")
    first = FakeGateway(
        [
            [ModelCompleted(message=RuntimeMessage(role="assistant", content=[ToolUseBlock(id="n", name="narrow", input={"value": "x"})]), stop_reason="tool_use")],
            [ModelCompleted(message=RuntimeMessage(role="assistant", content=[TextBlock(text="narrowed")]), stop_reason="end_turn")],
        ]
    )
    start = StartQueryParams(
        session_id="narrowed",
        model="saved",
        messages=[RuntimeMessage(role="user", content=[TextBlock(text="start")])],
        repository_root=str(tmp_path),
    )
    asyncio.run(collect(runtime(first, [EchoTool(), NarrowTool()], store), start))

    resumed = FakeGateway(
        [[ModelCompleted(message=RuntimeMessage(role="assistant", content=[TextBlock(text="done")]), stop_reason="end_turn")]]
    )
    asyncio.run(
        collect(
            runtime(resumed, [EchoTool(), NarrowTool()], store),
            ResumeQueryParams(session_id="narrowed", repository_root=str(tmp_path)),
        )
    )
    assert [tool["name"] for tool in resumed.requests[0].tools] == ["echo"]


def test_cancelled_tool_gets_a_persisted_result_pair(tmp_path: Path) -> None:
    started = asyncio.Event()

    class BlockingTool(EchoTool):
        name = "blocking"

        async def call(self, input: EchoInput, context: ToolUseContext) -> ToolResult:
            started.set()
            await asyncio.Event().wait()
            return ToolResult(data={"value": input.value})

    store = FileSessionStore(tmp_path / "sessions")
    gateway = FakeGateway(
        [[ModelCompleted(message=RuntimeMessage(role="assistant", content=[ToolUseBlock(id="block-1", name="blocking", input={"value": "x"})]), stop_reason="tool_use")]]
    )

    async def scenario():
        task = asyncio.create_task(
            collect(
                runtime(gateway, [BlockingTool()], store),
                StartQueryParams(
                    session_id="cancelled",
                    model="test",
                    messages=[RuntimeMessage(role="user", content=[TextBlock(text="start")])],
                    repository_root=str(tmp_path),
                ),
            )
        )
        await started.wait()
        task.cancel()
        return await task

    events = asyncio.run(scenario())
    assert isinstance(events[-1], RunStopped)
    assert events[-1].transition.kind == "cancelled"
    snapshot = store.load("cancelled")
    assert snapshot is not None
    assert any(
        isinstance(block, ToolResultBlock) and block.tool_use_id == "block-1"
        for message in snapshot.messages
        for block in message.content
    )


def test_cancelled_model_request_marks_session_without_partial_message(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path / "sessions")

    class BlockingGateway:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.started.set()
            await asyncio.Event().wait()
            if False:
                yield ModelTextDelta(text="")

    gateway = BlockingGateway()
    agent = runtime(gateway, session_store=store)

    async def exercise() -> None:
        task = asyncio.create_task(
            collect(
                agent,
                StartQueryParams(
                    session_id="model-cancelled",
                    model="test",
                    repository_root=str(tmp_path),
                    messages=[
                        RuntimeMessage(
                            role="user",
                            content=[TextBlock(text="hello")],
                        )
                    ],
                ),
            )
        )
        await gateway.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    snapshot = store.load("model-cancelled")

    assert snapshot is not None
    assert snapshot.runtime_state.last_status == "cancelled"
    assert len(snapshot.messages) == 1
