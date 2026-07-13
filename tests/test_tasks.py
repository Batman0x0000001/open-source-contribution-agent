from __future__ import annotations

import json
import threading

from osc_agent.agent_loop import TOOLS, build_tool_handlers
from osc_agent.harness.tasks import (
    claim_task,
    complete_task,
    create_default_task_graph,
    create_task,
    get_task,
    list_tasks,
    load_all_tasks,
    load_task,
    tasks_dir,
)


def _task_id(output: str) -> str:
    return json.loads(output)["id"]


def test_create_task_writes_inspectable_json(tmp_path):
    output = create_task(
        repo_root=tmp_path,
        subject="repo scan",
        description="Read README and contribution guide.",
        files=["README.md"],
        evidence=["README exists"],
        worktree="main",
    )
    task_id = _task_id(output)
    path = tasks_dir(tmp_path) / f"{task_id}.json"

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["subject"] == "repo scan"
    assert data["status"] == "pending"
    assert data["files"] == ["README.md"]
    assert data["worktree"] == "main"


def test_task_persists_and_can_be_loaded_after_restart(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="plan"))

    restored = load_task(tmp_path, task_id)

    assert restored.id == task_id
    assert restored.subject == "plan"


def test_blocked_task_cannot_be_claimed_until_dependency_completed(tmp_path):
    dependency_id = _task_id(create_task(repo_root=tmp_path, subject="schema"))
    dependent_id = _task_id(create_task(repo_root=tmp_path, subject="api", blockedBy=[dependency_id]))

    blocked = claim_task(repo_root=tmp_path, task_id=dependent_id, owner="agent-a")
    claimed_dependency = claim_task(repo_root=tmp_path, task_id=dependency_id, owner="agent-a")
    completed = complete_task(repo_root=tmp_path, task_id=dependency_id)
    claimed_dependent = claim_task(repo_root=tmp_path, task_id=dependent_id, owner="agent-b")

    assert "blocked by" in blocked
    assert claimed_dependency.startswith("Claimed")
    assert "Unblocked: api" in completed
    assert claimed_dependent.startswith("Claimed")
    assert load_task(tmp_path, dependent_id).owner == "agent-b"


def test_claim_rejects_non_pending_task(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="edit"))
    claim_task(repo_root=tmp_path, task_id=task_id, owner="agent")

    result = claim_task(repo_root=tmp_path, task_id=task_id, owner="other")

    assert "is in_progress, cannot claim" in result


def test_complete_task_records_evidence(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="test"))
    claim_task(repo_root=tmp_path, task_id=task_id)

    result = complete_task(repo_root=tmp_path, task_id=task_id, evidence=["pytest passed"])
    task = load_task(tmp_path, task_id)

    assert result.startswith("Completed")
    assert task.status == "completed"
    assert task.evidence == ["pytest passed"]


def test_complete_task_requires_claim_and_matching_owner(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="test"))

    assert "is pending, cannot complete" in complete_task(repo_root=tmp_path, task_id=task_id)
    claim_task(repo_root=tmp_path, task_id=task_id, owner="alice")
    assert "owned by alice" in complete_task(repo_root=tmp_path, task_id=task_id, owner="bob")
    assert complete_task(repo_root=tmp_path, task_id=task_id, owner="alice").startswith("Completed")


def test_concurrent_claim_has_single_winner(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="edit"))
    barrier = threading.Barrier(3)
    results: list[str] = []

    def claim(owner: str) -> None:
        barrier.wait()
        results.append(claim_task(repo_root=tmp_path, task_id=task_id, owner=owner))

    threads = [threading.Thread(target=claim, args=(owner,)) for owner in ("alice", "bob")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sum(result.startswith("Claimed") for result in results) == 1


def test_list_and_get_task_return_readable_state(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="summarize", description="Write final notes."))

    listing = list_tasks(repo_root=tmp_path)
    detail = get_task(repo_root=tmp_path, task_id=task_id)

    assert task_id in listing
    assert "[pending] summarize" in listing
    assert json.loads(detail)["description"] == "Write final notes."


def test_create_default_task_graph_has_ordered_dependencies(tmp_path):
    created = create_default_task_graph(tmp_path)

    assert [task.subject for task in created] == ["repo scan", "plan", "edit", "test", "summarize", "draft PR"]
    assert len(load_all_tasks(tmp_path)) == 6
    assert created[0].blockedBy == []
    assert created[-1].blockedBy == [created[-2].id]


def test_agent_loop_registers_task_tools(tmp_path):
    tool_names = {tool["name"] for tool in TOOLS}
    handlers = build_tool_handlers(tmp_path)

    assert {"create_task", "list_tasks", "get_task", "claim_task", "complete_task"} <= tool_names
    output = handlers["create_task"](subject="repo scan")
    task_id = _task_id(output)
    assert task_id in handlers["list_tasks"]()
    assert "Claimed" in handlers["claim_task"](task_id=task_id, owner="agent")
