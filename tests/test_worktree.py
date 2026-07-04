from __future__ import annotations

import json
import subprocess

from osc_agent.agent_loop import TOOLS, build_tool_handlers
from osc_agent.harness.tasks import create_task, load_task
from osc_agent.harness.teams import _teammate_handlers, claim_next_available_task
from osc_agent.harness.worktree import (
    bind_task_to_worktree,
    create_worktree,
    events_path,
    keep_worktree,
    remove_worktree,
    validate_worktree_name,
    worktree_path,
)


def _task_id(output: str) -> str:
    return json.loads(output)["id"]


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(path):
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
    (path / "README.md").write_text("root\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "init")


def test_validate_worktree_name_rejects_unsafe_values():
    assert validate_worktree_name("feature_1.2-3") is None
    assert validate_worktree_name("../escape") is not None
    assert validate_worktree_name("bad/name") is not None
    assert validate_worktree_name("..") is not None


def test_create_worktree_binds_task_and_logs_event(tmp_path):
    _init_repo(tmp_path)
    task_id = _task_id(create_task(repo_root=tmp_path, subject="isolated edit"))

    result = create_worktree(repo_root=tmp_path, name="isolated-edit", task_id=task_id)
    task = load_task(tmp_path, task_id)
    events = events_path(tmp_path).read_text(encoding="utf-8")

    assert result.startswith("Worktree 'isolated-edit' created")
    assert worktree_path(tmp_path, "isolated-edit").exists()
    assert task.worktree == "isolated-edit"
    assert '"type": "create"' in events


def test_keep_worktree_logs_event(tmp_path):
    _init_repo(tmp_path)
    create_worktree(repo_root=tmp_path, name="keep-me")

    result = keep_worktree(repo_root=tmp_path, name="keep-me")

    assert "kept for review" in result
    assert '"type": "keep"' in events_path(tmp_path).read_text(encoding="utf-8")


def test_remove_worktree_protects_dirty_state(tmp_path):
    _init_repo(tmp_path)
    create_worktree(repo_root=tmp_path, name="dirty")
    (worktree_path(tmp_path, "dirty") / "dirty.txt").write_text("change\n", encoding="utf-8")

    protected = remove_worktree(repo_root=tmp_path, name="dirty")
    removed = remove_worktree(repo_root=tmp_path, name="dirty", discard_changes=True)

    assert "uncommitted changes" in protected
    assert removed == "Removed worktree 'dirty'"
    assert not worktree_path(tmp_path, "dirty").exists()


def test_bind_task_to_worktree_does_not_claim_task(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="bind only"))
    worktree_path(tmp_path, "manual").mkdir(parents=True)

    result = bind_task_to_worktree(repo_root=tmp_path, task_id=task_id, worktree="manual")
    task = load_task(tmp_path, task_id)

    assert result == f"Bound task {task_id} to worktree manual"
    assert task.status == "pending"
    assert task.worktree == "manual"


def test_teammate_claim_switches_tool_cwd_to_bound_worktree(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="read isolated"))
    isolated = worktree_path(tmp_path, "manual")
    isolated.mkdir(parents=True)
    (isolated / "note.txt").write_text("from worktree", encoding="utf-8")
    bind_task_to_worktree(repo_root=tmp_path, task_id=task_id, worktree="manual")
    context = {"path": None}
    handlers = _teammate_handlers(tmp_path, name="alice", allow_write=False, worktree_context=context)

    claim = handlers["claim_task"](task_id=task_id)
    content = handlers["read_file"](path="note.txt")

    assert claim.startswith("Claimed")
    assert context["path"] == isolated
    assert content == "from worktree"


def test_auto_claim_sets_worktree_context(tmp_path):
    task_id = _task_id(create_task(repo_root=tmp_path, subject="auto isolated"))
    isolated = worktree_path(tmp_path, "auto")
    isolated.mkdir(parents=True)
    bind_task_to_worktree(repo_root=tmp_path, task_id=task_id, worktree="auto")
    context = {"path": None}

    message = claim_next_available_task(repo_root=tmp_path, owner="alice", worktree_context=context)

    assert "[Auto-claimed task]" in message
    assert context["path"] == isolated


def test_agent_loop_registers_worktree_tools(tmp_path):
    tool_names = {tool["name"] for tool in TOOLS}
    handlers = build_tool_handlers(tmp_path)

    assert {"create_worktree", "keep_worktree", "remove_worktree"} <= tool_names
    assert "worktree name" in handlers["create_worktree"](name="../bad")
