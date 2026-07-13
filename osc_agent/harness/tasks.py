"""
创建 / 加载任务请求
    ↓
定位 .osc_agent/tasks 目录
    ↓
读取或创建 task JSON 文件
    ↓
构建 ContributionTask 数据结构
    ↓
写入 / 更新 task 状态（pending / in_progress / completed）
    ↓
（可选）检查 blockedBy 依赖
    ↓
更新任务依赖状态
    ↓
保存 JSON 到磁盘
    ↓
返回任务信息 / 状态文本
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

CONTRIBUTION_TASK_STATUSES = {"pending", "in_progress", "completed"}
_task_lock = threading.RLock()

CONTRIBUTION_TASK_TOOLS = [
    {
        "name": "create_task",
        "description": "Create a persistent contribution task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "blockedBy": {"type": "array", "items": {"type": "string"}},
                "files": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "worktree": {"type": "string"},
            },
            "required": ["subject"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_tasks",
        "description": "List persistent contribution tasks and statuses.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_task",
        "description": "Read one persistent contribution task as JSON.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "claim_task",
        "description": "Claim an unblocked pending contribution task.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}, "owner": {"type": "string"}},
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "complete_task",
        "description": "Mark an in-progress contribution task completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ContributionTask:
    id: str
    subject: str
    description: str = ""
    status: str = "pending"
    owner: str | None = None
    blockedBy: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    worktree: str | None = None


def tasks_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "tasks"


def ensure_tasks_dir(repo_root: Path) -> Path:
    directory = tasks_dir(repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def task_path(repo_root: Path, task_id: str) -> Path:
    safe_id = _safe_task_id(task_id)
    return ensure_tasks_dir(repo_root) / f"{safe_id}.json"


def create_task(
    *,
    repo_root: Path,
    subject: str,
    description: str = "",
    blockedBy: list[str] | None = None,
    files: list[str] | None = None,
    evidence: list[str] | None = None,
    worktree: str | None = None,
) -> str:
    """创建一个可跨会话恢复的任务 JSON 文件。"""
    if not subject.strip():
        raise ValueError("subject must be a non-empty string")
    task = ContributionTask(
        id=_new_task_id(),
        subject=subject.strip(),
        description=description.strip(),
        blockedBy=list(blockedBy or []),
        files=list(files or []),
        evidence=list(evidence or []),
        worktree=worktree,
    )
    save_task(repo_root, task)
    return _format_task_json(task)


def create_default_task_graph(repo_root: Path) -> list[ContributionTask]:
    """生成标准贡献流程模板：scan -> plan -> edit -> test -> summarize -> draft PR。"""
    subjects = [
        ("repo scan", "Inspect repository structure and contribution guidance.", []),
        ("plan", "Create the implementation plan.", ["repo scan"]),
        ("edit", "Make the scoped code or documentation changes.", ["plan"]),
        ("test", "Run focused verification.", ["edit"]),
        ("summarize", "Summarize modified files, tests, and risks.", ["test"]),
        ("draft PR", "Draft a concise PR title and body.", ["summarize"]),
    ]
    created: list[ContributionTask] = []
    ids_by_subject: dict[str, str] = {}
    with _task_lock:
        for subject, description, deps in subjects:
            task = ContributionTask(
                id=_new_task_id(),
                subject=subject,
                description=description,
                blockedBy=[ids_by_subject[dep] for dep in deps],
            )
            save_task(repo_root, task)
            ids_by_subject[subject] = task.id
            created.append(task)
    return created


def list_tasks(*, repo_root: Path) -> str:
    tasks = load_all_tasks(repo_root)
    if not tasks:
        return "(no tasks)"
    lines = []
    for task in tasks:
        deps = f" blockedBy={task.blockedBy}" if task.blockedBy else ""
        owner = f" owner={task.owner}" if task.owner else ""
        lines.append(f"- {task.id} [{task.status}] {task.subject}{owner}{deps}")
    return "\n".join(lines)


def get_task(*, repo_root: Path, task_id: str) -> str:
    return _format_task_json(load_task(repo_root, task_id))


def claim_task(*, repo_root: Path, task_id: str, owner: str = "agent") -> str:
    with _task_lock:
        task = load_task(repo_root, task_id)
        if task.status != "pending":
            return f"Task {task.id} is {task.status}, cannot claim"
        if task.owner:
            return f"Task {task.id} already owned by {task.owner}"

        blocked_by = blocking_dependencies(repo_root, task)
        if blocked_by:
            return f"Task {task.id} is blocked by: {', '.join(blocked_by)}"

        task.status = "in_progress"
        task.owner = owner or "agent"
        save_task(repo_root, task)
        return f"Claimed {task.id} ({task.subject})"


def complete_task(
    *,
    repo_root: Path,
    task_id: str,
    evidence: list[str] | None = None,
    owner: str | None = None,
) -> str:
    with _task_lock:
        task = load_task(repo_root, task_id)
        if task.status != "in_progress":
            return f"Task {task.id} is {task.status}, cannot complete"
        if owner is not None and task.owner != owner:
            return f"Task {task.id} is owned by {task.owner}, not {owner}"
        task.status = "completed"
        if evidence:
            task.evidence.extend(evidence)
        save_task(repo_root, task)

        unblocked = [
            candidate.subject
            for candidate in load_all_tasks(repo_root)
            if candidate.status == "pending" and candidate.blockedBy and not blocking_dependencies(repo_root, candidate)
        ]
        message = f"Completed {task.id} ({task.subject})"
        if unblocked:
            message += "\nUnblocked: " + ", ".join(unblocked)
        return message


def load_task(repo_root: Path, task_id: str) -> ContributionTask:
    with _task_lock:
        path = task_path(repo_root, task_id)
        if not path.exists():
            raise ValueError(f"task not found: {task_id}")
        return ContributionTask(**json.loads(path.read_text(encoding="utf-8")))


def load_all_tasks(repo_root: Path) -> list[ContributionTask]:
    with _task_lock:
        directory = ensure_tasks_dir(repo_root)
        tasks = [
            ContributionTask(**json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(directory.glob("*.json"))
        ]
        return sorted(tasks, key=lambda task: task.id)


def save_task(repo_root: Path, task: ContributionTask) -> Path:
    if task.status not in CONTRIBUTION_TASK_STATUSES:
        raise ValueError(f"invalid task status: {task.status}")
    with _task_lock:
        path = task_path(repo_root, task.id)
        temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            temp.write_text(json.dumps(asdict(task), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        return path


def blocking_dependencies(repo_root: Path, task: ContributionTask) -> list[str]:
    blocked: list[str] = []
    for dep_id in task.blockedBy:
        try:
            dep = load_task(repo_root, dep_id)
        except ValueError:
            blocked.append(dep_id)
            continue
        if dep.status != "completed":
            blocked.append(dep_id)
    return blocked


def _format_task_json(task: ContributionTask) -> str:
    return json.dumps(asdict(task), ensure_ascii=False, indent=2)


def _new_task_id() -> str:
    return f"task_{int(time.time() * 1000)}_{secrets.token_hex(3)}"


def _safe_task_id(task_id: str) -> str:
    cleaned = "".join(ch for ch in task_id if ch.isalnum() or ch in {"_", "-"})
    if not cleaned or cleaned != task_id:
        raise ValueError("task_id may contain only letters, numbers, underscores, or hyphens")
    return cleaned
