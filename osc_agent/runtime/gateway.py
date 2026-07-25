from __future__ import annotations

from typing import Annotated, AsyncIterator, Literal, Protocol, TypeAlias

from pydantic import Field, JsonValue

from osc_agent.runtime.models import ContractModel, FrozenContractModel, RuntimeMessage


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
    ModelTextDelta | ModelCompleted,
    Field(discriminator="type"),
]


class ModelGateway(Protocol):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...


class ModelGatewayError(Exception):
    def __init__(self, *, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
