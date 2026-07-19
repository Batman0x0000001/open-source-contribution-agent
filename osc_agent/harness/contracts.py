from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_BUDGET = "FAILED_BUDGET"
    FAILED_TOOL = "FAILED_TOOL"
    BLOCKED_NEEDS_USER = "BLOCKED_NEEDS_USER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    STALE_RUN = "STALE_RUN"


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


def action_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    normalized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(f"{name}:{normalized}".encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    summary: str
    error_code: str | None = None
    retryable: bool = False
    side_effect: bool = False
    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    fingerprint: str | None = None
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_model_content(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


def normalize_tool_result(
    value: Any,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str | None,
    latency_ms: int,
    side_effect: bool,
) -> ToolResult:
    if isinstance(value, ToolResult):
        return ToolResult(
            **{
                **value.to_dict(),
                "call_id": value.call_id or call_id,
                "fingerprint": value.fingerprint or action_fingerprint(tool_name, arguments),
                "latency_ms": value.latency_ms if value.latency_ms is not None else latency_ms,
                "side_effect": value.side_effect or side_effect,
            }
        )

    summary = str(value)
    lowered = summary.lower()
    failed = (
        lowered.startswith("error:")
        or lowered.startswith("permission denied:")
        or lowered.startswith("permission required:")
        or lowered.startswith("git error:")
    )
    error_code = None
    retryable = False
    if failed:
        error_code = "TOOL_ERROR"
        if "timeout" in lowered:
            error_code = "COMMAND_TIMEOUT"
            retryable = True
        elif lowered.startswith("permission required:"):
            error_code = "PERMISSION_REQUIRED"
        elif "permission" in lowered:
            error_code = "PERMISSION_DENIED"
        elif "not found" in lowered:
            error_code = "NOT_FOUND"
        elif "nonzero_exit" in lowered:
            error_code = "NONZERO_EXIT"

    return ToolResult(
        ok=not failed,
        error_code=error_code,
        retryable=retryable,
        side_effect=side_effect,
        summary=summary,
        call_id=call_id,
        fingerprint=action_fingerprint(tool_name, arguments),
        latency_ms=latency_ms,
    )


@dataclass
class RunMetrics:
    model_calls: int = 0
    model_attempts: int = 0
    fallback_switches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    retries: int = 0
    elapsed_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {**asdict(self), "total_tokens": self.total_tokens}


@dataclass(frozen=True)
class AgentRunResult:
    status: RunStatus
    response: Any | None
    reason: str
    metrics: RunMetrics

    @property
    def stop_reason(self) -> Any:
        return getattr(self.response, "stop_reason", None)

    @property
    def content(self) -> Any:
        return getattr(self.response, "content", [])
