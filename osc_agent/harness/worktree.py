"""
LLM / Agent 调用 worktree tool
    ↓
校验 worktree name 是否安全
    ↓
在 .osc_agent/worktrees 下创建 / 保留 / 删除独立 worktree
    ↓
可选：把 task_id 绑定到 task.worktree
    ↓
记录 events.jsonl + append_trace
    ↓
队友认领任务后通过 resolve_task_worktree 找到隔离工作目录
    ↓
在独立 worktree 中执行任务，降低并行编辑冲突
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from time import time

from osc_agent.harness.tasks import load_task, save_task
from osc_agent.harness.trace import append_trace

WORKTREE_TOOLS = [
    {
        "name": "create_worktree",
        "description": "Create a git worktree under .osc_agent/worktrees and optionally bind it to a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "keep_worktree",
        "description": "Keep a worktree for later review and log the decision.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "remove_worktree",
        "description": "Remove a worktree. Dirty worktrees are protected unless discard_changes is true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "discard_changes": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
]


def validate_worktree_name(name: str) -> str | None:
    """只允许安全 slug，防止路径穿越或创建奇怪分支名。"""
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name):
        return "worktree name must match [A-Za-z0-9._-]{1,64}"
    if name in {".", ".."}:
        return "worktree name cannot be . or .."
    return None


def worktrees_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "worktrees"


def worktree_path(repo_root: Path, name: str) -> Path:
    error = validate_worktree_name(name)
    if error:
        raise ValueError(error)
    return worktrees_dir(repo_root) / name


def create_worktree(*, repo_root: Path, name: str, task_id: str = "") -> str:
    """创建独立 git worktree；只有 git 成功后才绑定任务和记录事件。"""
    error = validate_worktree_name(name)
    if error:
        return f"Error: {error}"
    target = worktree_path(repo_root, name)
    if target.exists():
        return f"Error: worktree {name} already exists"
    target.parent.mkdir(parents=True, exist_ok=True)

    branch = _branch_name(name)
    ok, output = _run_git(repo_root, ["worktree", "add", str(target), "-b", branch, "HEAD"])
    if not ok:
        return f"Git error: {output}"
    if task_id:
        bound = bind_task_to_worktree(repo_root=repo_root, task_id=task_id, worktree=name)
        if bound.startswith("Error:"):
            return bound
    _log_event(repo_root, "create", name, task_id)
    append_trace(repo_root, "worktree_create", {"worktree": name, "task_id": task_id, "path": str(target)})
    return f"Worktree '{name}' created at {target}"


def bind_task_to_worktree(*, repo_root: Path, task_id: str, worktree: str) -> str:
    """只写 task.worktree 字段，不改变任务状态，等待队友自行 claim。"""
    error = validate_worktree_name(worktree)
    if error:
        return f"Error: {error}"
    try:
        task = load_task(repo_root, task_id)
    except ValueError as exc:
        return f"Error: {exc}"
    task.worktree = worktree
    save_task(repo_root, task)
    append_trace(repo_root, "worktree_bind", {"worktree": worktree, "task_id": task_id})
    return f"Bound task {task_id} to worktree {worktree}"


def keep_worktree(*, repo_root: Path, name: str) -> str:
    error = validate_worktree_name(name)
    if error:
        return f"Error: {error}"
    target = worktree_path(repo_root, name)
    if not target.exists():
        return f"Error: worktree {name} does not exist"
    _log_event(repo_root, "keep", name)
    append_trace(repo_root, "worktree_keep", {"worktree": name})
    return f"Worktree '{name}' kept for review (branch: {_branch_name(name)})"


def remove_worktree(*, repo_root: Path, name: str, discard_changes: bool = False) -> str:
    """删除前默认检查脏状态，避免静默丢弃队友产物。"""
    error = validate_worktree_name(name)
    if error:
        return f"Error: {error}"
    target = worktree_path(repo_root, name)
    if not target.exists():
        return f"Error: worktree {name} does not exist"
    dirty = _dirty_status(target)
    if dirty and not discard_changes:
        return "Error: worktree has uncommitted changes; use discard_changes=true or keep_worktree"

    args = ["worktree", "remove", str(target)]
    if discard_changes:
        args.append("--force")
    ok, output = _run_git(repo_root, args)
    if not ok:
        return f"Git error: {output}"
    _run_git(repo_root, ["branch", "-D", _branch_name(name)])
    _log_event(repo_root, "remove", name)
    append_trace(repo_root, "worktree_remove", {"worktree": name, "discard_changes": discard_changes})
    return f"Removed worktree '{name}'"


def resolve_task_worktree(repo_root: Path, task_id: str) -> Path | None:
    """队友认领任务后用它找到隔离 cwd；未绑定或目录不存在则返回 None。"""
    try:
        task = load_task(repo_root, task_id)
    except ValueError:
        return None
    if not task.worktree:
        return None
    target = worktree_path(repo_root, task.worktree)
    return target if target.exists() else None


def events_path(repo_root: Path) -> Path:
    return worktrees_dir(repo_root) / "events.jsonl"


def _branch_name(name: str) -> str:
    return f"wt/{name}"


def _run_git(repo_root: Path, args: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return completed.returncode == 0, output or "(no output)"


def _dirty_status(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return (completed.stdout or "").strip()


def _log_event(repo_root: Path, event_type: str, worktree: str, task_id: str = "") -> None:
    path = events_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"type": event_type, "worktree": worktree, "task_id": task_id, "ts": time()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
