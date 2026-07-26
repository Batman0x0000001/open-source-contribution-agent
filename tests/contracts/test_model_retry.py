from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import (
    ModelCompleted,
    ModelEvent,
    ModelGatewayError,
    ModelRequest,
    ModelTextDelta,
    RetryingModelGateway,
    RetryPolicy,
)
from osc_agent.runtime.models import (
    ModelRetryScheduled,
    RuntimeMessage,
    StartQueryParams,
    TextBlock,
)
from osc_agent.runtime.query import AgentRuntime
from osc_agent.runtime.session_store import FileSessionStore
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor


class SequenceGateway:
    def __init__(self, turns: list[list[ModelEvent] | ModelGatewayError]) -> None:
        self.turns = list(turns)
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        turn = self.turns.pop(0)
        if isinstance(turn, ModelGatewayError):
            raise turn
        for event in turn:
            if isinstance(event, ModelGatewayError):
                raise event
            yield event


def request() -> ModelRequest:
    return ModelRequest(
        model="test",
        messages=[RuntimeMessage(role="user", content=[TextBlock(text="hello")])],
        max_output_tokens=10,
    )


async def collect(gateway: RetryingModelGateway):
    return [event async for event in gateway.stream(request())]


def test_retry_replays_request_after_partial_output_without_hiding_notice() -> None:
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    gateway = SequenceGateway(
        [
            [
                ModelTextDelta(text="partial"),
                ModelGatewayError(
                    code="MODEL_RATE_LIMITED",
                    message="retry",
                    retryable=True,
                    retry_after_seconds=2,
                ),
            ],
            [
                ModelTextDelta(text="complete"),
                ModelCompleted(
                    message=RuntimeMessage(
                        role="assistant",
                        content=[TextBlock(text="complete")],
                    ),
                    stop_reason="end_turn",
                ),
            ],
        ]
    )
    retrying = RetryingModelGateway(
        gateway,
        RetryPolicy(max_attempts=3, base_seconds=1, max_seconds=8),
        sleep=sleep,
        jitter=lambda _low, _high: 0,
    )

    events = asyncio.run(collect(retrying))

    notice = next(event for event in events if isinstance(event, ModelRetryScheduled))
    assert notice.attempt == 2
    assert notice.had_partial_output is True
    assert sleeps == [2]
    assert gateway.calls == 2
    assert isinstance(events[-1], ModelCompleted)


def test_non_retryable_model_error_fails_immediately() -> None:
    gateway = SequenceGateway(
        [
            ModelGatewayError(
                code="MODEL_AUTHENTICATION_FAILED",
                message="invalid",
                retryable=False,
            )
        ]
    )
    retrying = RetryingModelGateway(gateway, RetryPolicy(max_attempts=3))

    with pytest.raises(ModelGatewayError, match="invalid"):
        asyncio.run(collect(retrying))

    assert gateway.calls == 1


def test_context_length_error_is_left_for_runtime_compaction() -> None:
    gateway = SequenceGateway(
        [
            ModelGatewayError(
                code="CONTEXT_LENGTH_EXCEEDED",
                message="large",
                retryable=True,
            )
        ]
    )
    retrying = RetryingModelGateway(gateway, RetryPolicy(max_attempts=3))

    with pytest.raises(ModelGatewayError, match="large"):
        asyncio.run(collect(retrying))

    assert gateway.calls == 1


def test_runtime_exposes_retry_event_but_persists_only_completed_message(
    tmp_path,
) -> None:
    raw = SequenceGateway(
        [
            [
                ModelTextDelta(text="discarded partial"),
                ModelGatewayError(
                    code="MODEL_CONNECTION_FAILED",
                    message="temporary",
                    retryable=True,
                ),
            ],
            [
                ModelCompleted(
                    message=RuntimeMessage(
                        role="assistant",
                        content=[TextBlock(text="final")],
                    ),
                    stop_reason="end_turn",
                )
            ],
        ]
    )
    gateway = RetryingModelGateway(
        raw,
        RetryPolicy(max_attempts=2, base_seconds=0, max_seconds=0),
    )
    registry = ToolRegistry()
    store = FileSessionStore(tmp_path / "sessions")
    runtime = AgentRuntime(
        QueryDependencies(
            model_gateway=gateway,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            session_store=store,
        )
    )

    async def run():
        return [
            event
            async for event in runtime.query(
                StartQueryParams(
                    session_id="retry-session",
                    model="test",
                    repository_root=str(tmp_path),
                    messages=[
                        RuntimeMessage(
                            role="user",
                            content=[TextBlock(text="hello")],
                        )
                    ],
                )
            )
        ]

    events = asyncio.run(run())
    snapshot = store.load("retry-session")

    assert any(isinstance(event, ModelRetryScheduled) for event in events)
    assert snapshot is not None
    assert len(snapshot.messages) == 2
    assert snapshot.messages[-1].content[0].text == "final"
