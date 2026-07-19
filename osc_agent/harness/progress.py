from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from osc_agent.harness.contracts import RunStatus, ToolResult, action_fingerprint


@dataclass(frozen=True)
class ProgressStop:
    status: RunStatus
    reason: str


@dataclass
class ProgressGuard:
    """集中维护工具重复、连续失败和无进展判定。"""

    repeat_action_limit: int
    consecutive_failure_limit: int
    no_progress_limit: int
    previous_fingerprint: str | None = None
    repeated_actions: int = 0
    consecutive_failures: int = 0
    seen_fingerprints: set[str] = field(default_factory=set)
    no_progress_calls: int = 0

    def before_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, ProgressStop | None]:
        fingerprint = action_fingerprint(tool_name, arguments)
        if fingerprint == self.previous_fingerprint:
            self.repeated_actions += 1
        else:
            self.previous_fingerprint = fingerprint
            self.repeated_actions = 1

        if self.repeated_actions >= self.repeat_action_limit:
            return fingerprint, ProgressStop(
                RunStatus.BLOCKED_NEEDS_USER,
                f"tool action repeated {self.repeated_actions} times: {tool_name}",
            )
        return fingerprint, None

    def after_tool(self, fingerprint: str, result: ToolResult) -> ProgressStop | None:
        if result.ok:
            self.consecutive_failures = 0
            if fingerprint in self.seen_fingerprints and not result.side_effect:
                self.no_progress_calls += 1
            else:
                self.no_progress_calls = 0
            self.seen_fingerprints.add(fingerprint)
        else:
            self.consecutive_failures += 1
            self.no_progress_calls += 1

        if self.consecutive_failures >= self.consecutive_failure_limit:
            return ProgressStop(
                RunStatus.FAILED_TOOL,
                f"{self.consecutive_failures} consecutive tool failures",
            )
        if self.no_progress_calls >= self.no_progress_limit:
            return ProgressStop(
                RunStatus.BLOCKED_NEEDS_USER,
                f"no new evidence or state change after {self.no_progress_calls} tool calls",
            )
        return None
