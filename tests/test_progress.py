from __future__ import annotations

from osc_agent.harness.contracts import RunStatus, ToolResult
from osc_agent.harness.progress import ProgressGuard


def test_progress_guard_stops_repeated_tool_calls():
    guard = ProgressGuard(repeat_action_limit=2, consecutive_failure_limit=3, no_progress_limit=3)

    fingerprint, first_stop = guard.before_tool("read_file", {"path": "a.py"})
    _, second_stop = guard.before_tool("read_file", {"path": "a.py"})

    assert fingerprint
    assert first_stop is None
    assert second_stop is not None
    assert second_stop.status is RunStatus.BLOCKED_NEEDS_USER


def test_progress_guard_tracks_failures_and_no_progress():
    guard = ProgressGuard(repeat_action_limit=4, consecutive_failure_limit=2, no_progress_limit=3)
    failure = ToolResult(ok=False, summary="Error: missing")

    first = guard.after_tool("one", failure)
    second = guard.after_tool("two", failure)

    assert first is None
    assert second is not None
    assert second.status is RunStatus.FAILED_TOOL


def test_progress_guard_ignores_repeated_side_effect_results():
    guard = ProgressGuard(repeat_action_limit=4, consecutive_failure_limit=3, no_progress_limit=2)
    result = ToolResult(ok=True, summary="done", side_effect=True)

    assert guard.after_tool("same", result) is None
    assert guard.after_tool("same", result) is None
    assert guard.no_progress_calls == 0
