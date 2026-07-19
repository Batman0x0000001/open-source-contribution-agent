"""
模型请求
    → with_retry() 执行一次逻辑请求
    → 对失败做结构化分类
    → 可重试错误按当前请求预算和截止时间退避
    → 连续 529 达到阈值时切换一次备用模型
    → 返回成功响应，或向 Agent Loop 抛出最终异常

本模块只负责请求失败恢复；Prompt 压缩、输出扩容和续写由 Agent Loop 处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import random
import time
from typing import Any, Callable

from osc_agent.harness.trace import append_trace, sanitize_trace_text

DEFAULT_MAX_TOKENS = 8_000
ESCALATED_MAX_TOKENS = 64_000
MAX_RETRIES_PER_REQUEST = 5
MAX_RETRY_DEADLINE_SECONDS = 120.0
MAX_SERVER_RETRY_AFTER_SECONDS = 60.0
MAX_CONTINUATIONS = 3
MAX_529_BEFORE_FALLBACK = 3
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 32.0
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
    "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces."
)


@dataclass
class ModelRequestRecoveryState:
    current_model: str
    fallback_model_id: str | None = None
    consecutive_529: int = 0
    retry_count: int = 0
    logical_request_count: int = 0
    total_attempts: int = 0
    fallback_switches: int = 0


@dataclass
class ResponseRecoveryState:
    max_tokens: int = DEFAULT_MAX_TOKENS
    has_escalated_tokens: bool = False
    continuation_count: int = 0
    attempted_reactive_compact: bool = False


def retry_delay(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None:
        return retry_after
    base = min(BASE_DELAY_SECONDS * (2 ** max(0, attempt)), MAX_DELAY_SECONDS)
    return base + random.uniform(0, base * 0.25)


def with_retry(
    create_fn: Callable[[str], Any],
    *,
    state: ModelRequestRecoveryState,
    repo_root: Path,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries_per_request: int = MAX_RETRIES_PER_REQUEST,
    retry_deadline_seconds: float = MAX_RETRY_DEADLINE_SECONDS,
    max_retry_after_seconds: float = MAX_SERVER_RETRY_AFTER_SECONDS,
) -> Any:
    """执行一次逻辑模型请求；每个请求独立计数，fallback 不重置当前请求的预算。"""
    state.logical_request_count += 1
    request_number = state.logical_request_count
    request_attempts = 0
    request_retries = 0
    started = time.monotonic()

    while True:
        attempted_model = state.current_model
        request_attempts += 1
        state.total_attempts += 1
        try:
            response = create_fn(attempted_model)
        except Exception as exc:
            error_code = classify_model_error(exc)
            if error_code != "OVERLOADED":
                state.consecutive_529 = 0
            error = safe_model_error(exc, error_code=error_code)

            if error_code == "PROMPT_TOO_LONG":
                _trace_request_finished(
                    repo_root,
                    state,
                    attempted_model=attempted_model,
                    request_number=request_number,
                    request_attempts=request_attempts,
                    request_retries=request_retries,
                    status="failed",
                    error=error,
                )
                raise

            if error_code not in {"RATE_LIMITED", "OVERLOADED", "NETWORK_ERROR"}:
                append_trace(
                    repo_root,
                    "model_error",
                    {**error, "model": attempted_model},
                )
                _trace_request_finished(
                    repo_root,
                    state,
                    attempted_model=attempted_model,
                    request_number=request_number,
                    request_attempts=request_attempts,
                    request_retries=request_retries,
                    status="failed",
                    error=error,
                )
                raise

            if request_retries >= max_retries_per_request:
                append_trace(
                    repo_root,
                    "model_retry_exhausted",
                    {
                        **error,
                        "model": attempted_model,
                        "retry_budget_per_request": max_retries_per_request,
                        "total_retries": state.retry_count,
                    },
                )
                _trace_request_finished(
                    repo_root,
                    state,
                    attempted_model=attempted_model,
                    request_number=request_number,
                    request_attempts=request_attempts,
                    request_retries=request_retries,
                    status="failed",
                    error=error,
                )
                raise

            if error_code == "OVERLOADED":
                state.consecutive_529 += 1
                if (
                    state.consecutive_529 >= MAX_529_BEFORE_FALLBACK
                    and state.fallback_model_id
                    and state.current_model != state.fallback_model_id
                ):
                    state.current_model = state.fallback_model_id
                    state.consecutive_529 = 0
                    state.fallback_switches += 1
                    append_trace(
                        repo_root,
                        "model_fallback",
                        {
                            "from_model": attempted_model,
                            "to_model": state.current_model,
                            "fallback_switches": state.fallback_switches,
                        },
                    )

            retry_after = _retry_after_seconds(exc)
            if retry_after is not None and retry_after > max_retry_after_seconds:
                append_trace(
                    repo_root,
                    "model_retry_exhausted",
                    {
                        **error,
                        "model": attempted_model,
                        "reason": "retry_after_exceeds_limit",
                        "retry_after_seconds": retry_after,
                        "max_retry_after_seconds": max_retry_after_seconds,
                    },
                )
                _trace_request_finished(
                    repo_root,
                    state,
                    attempted_model=attempted_model,
                    request_number=request_number,
                    request_attempts=request_attempts,
                    request_retries=request_retries,
                    status="failed",
                    error=error,
                )
                raise

            delay = retry_delay(request_retries, retry_after)
            remaining = retry_deadline_seconds - (time.monotonic() - started)
            if remaining <= 0 or delay > remaining:
                append_trace(
                    repo_root,
                    "model_retry_exhausted",
                    {
                        **error,
                        "model": attempted_model,
                        "reason": "retry_deadline",
                        "retry_after_seconds": delay,
                    },
                )
                _trace_request_finished(
                    repo_root,
                    state,
                    attempted_model=attempted_model,
                    request_number=request_number,
                    request_attempts=request_attempts,
                    request_retries=request_retries,
                    status="failed",
                    error=error,
                )
                raise

            append_trace(
                repo_root,
                "model_retry",
                {
                    **error,
                    "logical_request": request_number,
                    "request_attempt": request_attempts,
                    "delay_seconds": delay,
                    "failed_model": attempted_model,
                    "next_model": state.current_model,
                    "total_attempts": state.total_attempts,
                    "total_retries": state.retry_count + 1,
                },
            )
            sleep_fn(delay)
            request_retries += 1
            state.retry_count += 1
            continue

        state.consecutive_529 = 0
        _trace_request_finished(
            repo_root,
            state,
            attempted_model=attempted_model,
            request_number=request_number,
            request_attempts=request_attempts,
            request_retries=request_retries,
            status="succeeded",
        )
        return response


def safe_model_error(exc: Exception, *, error_code: str | None = None) -> dict[str, Any]:
    """只保留诊断所需字段，避免异常正文把凭据或超长响应写入 Trace。"""
    details: dict[str, Any] = {
        "error": sanitize_trace_text(exc, limit=1_000),
        "error_code": error_code or classify_model_error(exc),
        "exception_type": type(exc).__name__,
    }
    status_code = _status_code(exc)
    if status_code is not None:
        details["status_code"] = status_code
    request_id = _request_id(exc)
    if request_id:
        details["request_id"] = sanitize_trace_text(request_id, limit=200)
    return details


def _trace_request_finished(
    repo_root: Path,
    state: ModelRequestRecoveryState,
    *,
    attempted_model: str,
    request_number: int,
    request_attempts: int,
    request_retries: int,
    status: str,
    error: dict[str, Any] | None = None,
) -> None:
    append_trace(
        repo_root,
        "model_request_finished",
        {
            "logical_request": request_number,
            "status": status,
            "request_attempts": request_attempts,
            "request_retries": request_retries,
            "total_attempts": state.total_attempts,
            "total_retries": state.retry_count,
            # model 始终表示真实完成过调用的模型；selected_model 仅表示后续选择。
            "model": attempted_model,
            **(
                {"selected_model": state.current_model}
                if state.current_model != attempted_model
                else {}
            ),
            **({"error_code": error["error_code"]} if error else {}),
        },
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def is_prompt_too_long_error(exc: Exception) -> bool:
    error_type = _error_type(exc)
    text = str(exc).lower()
    return error_type == "prompt_too_long" or any(
        marker in text for marker in ("prompt_too_long", "prompt too long", "context length")
    )


def is_rate_limit_error(exc: Exception) -> bool:
    if _error_type(exc) == "rate_limit_error" or _status_code(exc) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def is_overloaded_error(exc: Exception) -> bool:
    if _error_type(exc) == "overloaded_error" or _status_code(exc) == 529:
        return True
    text = str(exc).lower()
    return "529" in text or "overloaded" in text or "overload" in text


def classify_model_error(exc: Exception) -> str:
    error_type = _error_type(exc)
    status_code = _status_code(exc)
    if error_type == "prompt_too_long":
        return "PROMPT_TOO_LONG"
    if error_type == "rate_limit_error" or status_code == 429:
        return "RATE_LIMITED"
    if error_type == "overloaded_error" or status_code == 529:
        return "OVERLOADED"
    if error_type in {"authentication_error", "permission_error"} or status_code in {401, 403}:
        return "AUTHENTICATION_ERROR"
    # Prompt Too Long 常以 invalid_request_error/400 返回，需在通用 400 分类前检查消息。
    if is_prompt_too_long_error(exc):
        return "PROMPT_TOO_LONG"
    if status_code == 400 or error_type == "invalid_request_error":
        return "INVALID_REQUEST"
    if status_code in {408, 409} or (status_code is not None and status_code >= 500):
        return "NETWORK_ERROR"

    text = str(exc).lower()
    class_name = type(exc).__name__.lower()
    if is_rate_limit_error(exc):
        return "RATE_LIMITED"
    if is_overloaded_error(exc):
        return "OVERLOADED"
    if any(marker in text for marker in ("401", "403", "authentication", "unauthorized", "api key")):
        return "AUTHENTICATION_ERROR"
    if any(marker in text for marker in ("400", "invalid request", "invalid_request")):
        return "INVALID_REQUEST"
    if any(marker in class_name for marker in ("connection", "timeout")) or any(
        marker in text for marker in ("connection", "network", "timeout", "temporarily unavailable")
    ):
        return "NETWORK_ERROR"
    return "UNKNOWN_MODEL_ERROR"


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _error_type(exc: Exception) -> str:
    value = getattr(exc, "type", None)
    if isinstance(value, str):
        return value.lower()
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            return str(error["type"]).lower()
    return ""


def _request_id(exc: Exception) -> str:
    value = getattr(exc, "request_id", None)
    if value:
        return str(value)
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return ""
    return str(headers.get("request-id") or headers.get("Request-Id") or "")
