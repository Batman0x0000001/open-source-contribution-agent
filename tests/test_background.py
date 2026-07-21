from __future__ import annotations

import json
import time

from osc_agent.agent_loop import TOOLS
from osc_agent.harness.background import (
    check_background_task,
    collect_background_results,
    is_slow_operation,
    should_run_background,
    start_background_task,
)


def test_background_requires_explicit_flag():
    tool_input = {"command": "python -m pytest"}

    assert is_slow_operation("bash", tool_input)
    assert not should_run_background("bash", tool_input)
    assert should_run_background("bash", {**tool_input, "run_in_background": True})


def test_background_task_writes_output_and_emits_notification(tmp_path):
    task_id = start_background_task(command="echo done", repo_root=tmp_path, runner=lambda: "done")

    for _ in range(50):
        detail = json.loads(check_background_task(task_id))
        if detail["status"] == "completed":
            break
        time.sleep(0.02)

    detail = json.loads(check_background_task(task_id))
    notifications = collect_background_results()

    assert detail["status"] == "completed"
    assert detail["output_preview"] == "done"
    assert len(notifications) == 1
    assert f"<task_id>{task_id}</task_id>" in notifications[0]
    assert "<summary>done</summary>" in notifications[0]
    assert collect_background_results() == []


def test_agent_loop_does_not_expose_background_check_tool():
    tool_names = {tool["name"] for tool in TOOLS}

    assert "check_background_task" not in tool_names


def test_background_notifications_are_filtered_by_repository(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    task_id = start_background_task(command="echo done", repo_root=other, runner=lambda: "done")
    for _ in range(50):
        if json.loads(check_background_task(task_id))["status"] == "completed":
            break
        time.sleep(0.02)

    assert collect_background_results(tmp_path) == []
    assert len(collect_background_results(other)) == 1
