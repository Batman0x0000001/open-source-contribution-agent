from __future__ import annotations

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import AsyncIterator

from anthropic import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
)

from osc_agent.runtime.gateway import (
    ModelCompleted,
    ModelEvent,
    ModelGatewayError,
    ModelRequest,
    ModelTextDelta,
)
from osc_agent.runtime.models import RuntimeMessage, TextBlock, ToolResultBlock, ToolUseBlock


class AnthropicModelGateway:
    def __init__(self, client: AsyncAnthropic) -> None:
        self.client = client

    @classmethod
    def from_config(cls, *, api_key: str, base_url: str | None = None) -> "AnthropicModelGateway":
        return cls(AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=0))

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        try:
            async with self.client.messages.stream(
                model=request.model,
                system=request.system_prompt,
                messages=[_message_param(message) for message in request.messages if message.role != "system"],
                tools=request.tools,
                max_tokens=request.max_output_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield ModelTextDelta(text=text)
                final = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - SDK 异常在 Provider 边界转成稳定契约。
            status_code = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            body_text = json.dumps(body, ensure_ascii=False, default=str) if body is not None else str(exc)
            is_context_error = status_code == 413 or any(
                marker in body_text.lower()
                for marker in ("context_length", "prompt is too long", "request_too_large")
            )
            code, retryable = _classify_error(exc, status_code, is_context_error)
            raise ModelGatewayError(
                code=code,
                message=str(exc) or type(exc).__name__,
                retryable=retryable,
                retry_after_seconds=_retry_after_seconds(exc),
            ) from exc

        blocks = []
        for block in final.content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )
        if not blocks:
            blocks.append(TextBlock(text=""))
        yield ModelCompleted(
            message=RuntimeMessage(role="assistant", content=blocks),
            stop_reason=final.stop_reason or "end_turn",
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )


def _message_param(message: RuntimeMessage) -> dict[str, object]:
    content: list[dict[str, object]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        elif isinstance(block, ToolResultBlock):
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": json.dumps(block.content, ensure_ascii=False),
                    "is_error": block.is_error,
                }
            )
    return {"role": message.role, "content": content}


def _classify_error(
    exc: Exception,
    status_code: int | None,
    is_context_error: bool,
) -> tuple[str, bool]:
    if is_context_error or isinstance(exc, RequestTooLargeError):
        return "CONTEXT_LENGTH_EXCEEDED", False
    if isinstance(exc, AuthenticationError):
        return "MODEL_AUTHENTICATION_FAILED", False
    if isinstance(exc, PermissionDeniedError):
        return "MODEL_PERMISSION_DENIED", False
    if isinstance(exc, NotFoundError):
        return "MODEL_NOT_FOUND", False
    if isinstance(exc, (BadRequestError, APIResponseValidationError)):
        return "MODEL_REQUEST_INVALID", False
    if isinstance(exc, RateLimitError) or status_code == 429:
        return "MODEL_RATE_LIMITED", True
    if isinstance(exc, OverloadedError) or status_code in {408, 409}:
        return "MODEL_OVERLOADED", True
    if isinstance(exc, APITimeoutError):
        return "MODEL_TIMEOUT", True
    if isinstance(exc, APIConnectionError):
        return "MODEL_CONNECTION_FAILED", True
    if status_code is not None and status_code >= 500:
        return "MODEL_SERVER_ERROR", True
    return "MODEL_REQUEST_FAILED", False


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            value = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(value, 0.0)
