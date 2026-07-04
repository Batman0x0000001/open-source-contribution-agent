"""
构造 LLM 请求
    ↓
调用 with_retry(LLM请求)
    ↓
LLM 正常返回？
        ├─ 是 → 进入 stop_reason 判断
        └─ 否 → 进入异常处理
                    ↓
        prompt too long？
            ├─ 是 → reactive_compact(messages)
            │         ↓
            │      continue（重新发请求）
            │
            └─ 否 → raise（向上抛出）
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from pathlib import Path
from typing import Any, Callable

from osc_agent.harness.trace import append_trace

DEFAULT_MAX_TOKENS = 8_000
ESCALATED_MAX_TOKENS = 64_000
MAX_RETRIES = 10
MAX_CONTINUATIONS = 3
MAX_529_BEFORE_FALLBACK = 3
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 32.0
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
    "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces."
)


@dataclass
class RecoveryState:
    current_model: str
    fallback_model_id: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    has_escalated_tokens: bool = False
    continuation_count: int = 0
    attempted_reactive_compact: bool = False
    consecutive_529: int = 0


def retry_delay(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None:
        return retry_after
    base = min(BASE_DELAY_SECONDS * (2 ** max(0, attempt)), MAX_DELAY_SECONDS)
    return base + random.uniform(0, base * 0.25)


def with_retry(
    create_fn: Callable[[str], Any],
    *,
    state: RecoveryState,
    repo_root: Path,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries: int = MAX_RETRIES,
) -> Any:
    """包装 LLM 调用：429/529 走指数退避，连续 529 可切 fallback model。"""
    attempt = 0
    while True:
        try:
            response = create_fn(state.current_model)
            state.consecutive_529 = 0
            return response
        except Exception as exc:
            if is_prompt_too_long_error(exc):
                raise
            if not (is_rate_limit_error(exc) or is_overloaded_error(exc)):
                append_trace(repo_root, "model_error", {"error": str(exc), "model": state.current_model})
                raise
            if attempt >= max_retries:
                append_trace(
                    repo_root,
                    "model_retry_exhausted",
                    {"error": str(exc), "attempts": attempt, "model": state.current_model},
                )
                raise

            if is_overloaded_error(exc):
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_529_BEFORE_FALLBACK and state.fallback_model_id:
                    state.current_model = state.fallback_model_id
                    state.consecutive_529 = 0
                    append_trace(repo_root, "model_fallback", {"model": state.current_model})

            delay = retry_delay(attempt)
            append_trace(
                repo_root,
                "model_retry",
                {
                    "attempt": attempt + 1,
                    "delay_seconds": delay,
                    "model": state.current_model,
                    "error": str(exc),
                },
            )
            sleep_fn(delay)
            attempt += 1


def is_prompt_too_long_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "prompt_too_long" in text or "prompt too long" in text or "context length" in text


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def is_overloaded_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "529" in text or "overloaded" in text or "overload" in text
