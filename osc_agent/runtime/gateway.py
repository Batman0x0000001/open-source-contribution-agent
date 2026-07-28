"""定义模型流式调用协议、事件和重试网关。"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Annotated, AsyncIterator, Literal, Protocol, TypeAlias

from pydantic import Field, JsonValue

from osc_agent.runtime.models import (
    ContractModel,
    FrozenContractModel,
    ModelRetryScheduled,
    RuntimeMessage,
)


class ModelRequest(FrozenContractModel):
    model: str = Field(min_length=1)
    system_prompt: str = ""
    messages: list[RuntimeMessage] = Field(min_length=1)
    tools: list[dict[str, JsonValue]] = Field(default_factory=list)
    max_output_tokens: int = Field(ge=1)


class ModelTextDelta(ContractModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ModelCompleted(ContractModel):
    type: Literal["completed"] = "completed"
    message: RuntimeMessage
    stop_reason: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


ModelEvent: TypeAlias = Annotated[
    ModelTextDelta | ModelCompleted | ModelRetryScheduled,
    Field(discriminator="type"),
]


class ModelGateway(Protocol):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...


class ModelGatewayError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class RetryPolicy(FrozenContractModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    base_seconds: float = Field(default=1, ge=0, le=60)
    max_seconds: float = Field(default=8, ge=0, le=300)


class RetryingModelGateway:
    """为所有模型消费者提供相同的有界重试，不缓存或修改请求。"""

    def __init__(
        self,
        gateway: ModelGateway,
        policy: RetryPolicy,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.gateway = gateway
        self.policy = policy
        self._sleep = sleep
        self._jitter = jitter

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        for attempt in range(1, self.policy.max_attempts + 1):
            had_partial_output = False
            completed = False
            failure: ModelGatewayError | None = None
            try:
                async for event in self.gateway.stream(request):
                    had_partial_output = had_partial_output or isinstance(
                        event,
                        ModelTextDelta,
                    )
                    completed = completed or isinstance(event, ModelCompleted)
                    yield event
            except ModelGatewayError as exc:
                failure = exc
            if completed:
                return
            if failure is None:
                failure = ModelGatewayError(
                    code="MODEL_STREAM_INCOMPLETE",
                    message="model stream ended without a completed message",
                    retryable=True,
                )
            if (
                not failure.retryable
                or failure.code == "CONTEXT_LENGTH_EXCEEDED"
                or attempt >= self.policy.max_attempts
            ):
                raise failure
            exponential = min(
                self.policy.max_seconds,
                self.policy.base_seconds * (2 ** (attempt - 1)),
            )
            delay = self._jitter(0, exponential) if exponential else 0.0
            if failure.retry_after_seconds is not None:
                delay = max(delay, failure.retry_after_seconds)
            delay = min(delay, self.policy.max_seconds)
            yield ModelRetryScheduled(
                attempt=attempt + 1,
                max_attempts=self.policy.max_attempts,
                delay_seconds=delay,
                error_code=failure.code,
                had_partial_output=had_partial_output,
            )
            await self._sleep(delay)
