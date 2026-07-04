from __future__ import annotations

import json

from osc_agent.harness import teams
from osc_agent.harness.protocols import request_shutdown
from osc_agent.harness.tasks import claim_task, create_task, load_task
from osc_agent.harness.teams import TEAMMATE_TOOLS, claim_next_available_task, idle_poll, scan_unclaimed_tasks


def _task_id(output: str) -> str:
    return json.loads(output)["id"]


def test_scan_unclaimed_tasks_skips_blocked_and_owned(tmp_path):
    ready_id = _task_id(create_task(repo_root=tmp_path, subject="ready"))
    owned_id = _task_id(create_task(repo_root=tmp_path, subject="owned"))
    blocked_id = _task_id(create_task(repo_root=tmp_path, subject="blocked", blockedBy=[ready_id]))
    claim_task(repo_root=tmp_path, task_id=owned_id, owner="alice")

    candidates = scan_unclaimed_tasks(tmp_path)

    assert [task["id"] for task in candidates] == [ready_id]
    assert blocked_id not in [task["id"] for task in candidates]


def test_claim_next_available_task_sets_owner_and_status(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="write tests"))

    message = claim_next_available_task(repo_root=tmp_path, owner="alice")
    task = load_task(tmp_path, task_id)

    assert "[Auto-claimed task]" in message
    assert task.status == "in_progress"
    assert task.owner == "alice"


def test_claim_task_rejects_existing_owner(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="owned"))
    claim_task(repo_root=tmp_path, task_id=task_id, owner="alice")
    task = load_task(tmp_path, task_id)
    task.status = "pending"
    from osc_agent.harness.tasks import save_task

    save_task(tmp_path, task)

    result = claim_task(repo_root=tmp_path, task_id=task_id, owner="bob")

    assert result == f"Task {task_id} already owned by alice"


def test_idle_poll_auto_claims_available_task(tmp_path, monkeypatch):
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT_SECONDS", 0.05)
    task_id = _task_id(create_task(repo_root=tmp_path, subject="scan repo"))
    messages: list[dict] = []

    result = idle_poll(repo_root=tmp_path, name="alice", messages=messages)

    assert result == "work"
    assert load_task(tmp_path, task_id).owner == "alice"
    assert "[Auto-claimed task]" in messages[-1]["content"]


def test_idle_poll_handles_shutdown_before_task_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT_SECONDS", 0.05)
    request_shutdown(repo_root=tmp_path, target="alice", reason="done")

    result = idle_poll(repo_root=tmp_path, name="alice", messages=[])

    assert result == "shutdown"


def test_teammate_tools_include_task_board_tools():
    assert {"list_tasks", "claim_task", "complete_task"} <= {tool["name"] for tool in TEAMMATE_TOOLS}
