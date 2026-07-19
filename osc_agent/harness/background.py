"""
agent 调用 bash tool
    ↓
判断是否为 slow command（提示用途）
    ↓
LLM 显式传入 run_in_background=true？
        ├─ 否 → 正常前台执行
        └─ 是 → start_background_task()
                    ↓
            启动 daemon thread 执行 runner
                    ↓
            写入 .osc_agent/background/task_id.out
                    ↓
            task 状态 = running
                    ↓
            执行完成/失败
                    ↓
            更新 task 状态 + finished_at
                    ↓
            append_trace 记录完成事件
                    ↓
            collect_background_results()
                    ↓
            将完成任务包装为 <task_notification>
                    ↓
            注入下一轮 agent loop
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from osc_agent.harness.trace import append_trace

SLOW_COMMAND_MARKERS = (
    "pytest",
    "npm test",
    "pnpm test",
    "yarn test",
    "cargo test",
    "cargo build",
    "npm run build",
    "pnpm build",
    "yarn build",
    "make",
    "compile",
    "docker build",
)

_lock = threading.Lock()
_counter = 0
_background_tasks: dict[str, "BackgroundTask"] = {}


@dataclass
class BackgroundTask:
    id: str
    command: str
    status: str
    started_at: str
    finished_at: str | None
    output_path: str
    repo_root: str
    notified: bool = False


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """识别适合后台运行的慢命令；这里只做提示，不自动改变执行方式。"""
    if tool_name != "bash":
        return False
    command = str(tool_input.get("command", "")).lower()
    return any(marker in command for marker in SLOW_COMMAND_MARKERS)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """后台执行必须由模型显式传入 run_in_background=true，避免误把前台命令异步化。"""
    return tool_name == "bash" and bool(tool_input.get("run_in_background"))


def start_background_task(*, command: str, repo_root: Path, runner: Callable[[], str]) -> str:
    """启动 daemon 线程执行命令，并把完整输出写到 .osc_agent/background。"""
    task_id = _next_task_id()
    output_path = _output_path(repo_root, task_id)
    task = BackgroundTask(
        id=task_id,
        command=command,
        status="running",
        started_at=_now(),
        finished_at=None,
        output_path=str(output_path),
        repo_root=str(repo_root.resolve()),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        _background_tasks[task_id] = task

    def worker() -> None:
        status = "completed"
        try:
            result = runner()
        except Exception as exc:  # pragma: no cover - defensive boundary for background threads
            status = "failed"
            result = "Error: " + json.dumps({"kind": "background_exception", "message": str(exc)}, ensure_ascii=False)

        output_path.write_text(str(result), encoding="utf-8")
        with _lock:
            current = _background_tasks[task_id]
            current.status = status
            current.finished_at = _now()
        append_trace(repo_root, "background_task_finished", {"task_id": task_id, "status": status, "command": command})

    threading.Thread(target=worker, daemon=True).start()
    append_trace(repo_root, "background_task_started", {"task_id": task_id, "command": command})
    return task_id


def check_background_task(task_id: str) -> str:
    """查询后台任务状态；任务完成后附带输出文件和短预览。"""
    with _lock:
        task = _background_tasks.get(task_id)
        if task is None:
            return f"Error: unknown background task {task_id}"
        payload = asdict(task)

    path = Path(payload["output_path"])
    if path.exists():
        payload["output_preview"] = path.read_text(encoding="utf-8")[:2_000]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def collect_background_results(repo_root: Path | None = None) -> list[str]:
    """收集已完成且尚未通知的任务，作为独立 task_notification 注入下一轮。"""
    ready: list[BackgroundTask] = []
    with _lock:
        for task in _background_tasks.values():
            matches_repo = repo_root is None or task.repo_root == str(repo_root.resolve())
            if matches_repo and task.status in {"completed", "failed"} and not task.notified:
                task.notified = True
                ready.append(BackgroundTask(**asdict(task)))

    notifications: list[str] = []
    for task in ready:
        output = Path(task.output_path).read_text(encoding="utf-8") if Path(task.output_path).exists() else ""
        notifications.append(
            "<task_notification>\n"
            f"  <task_id>{task.id}</task_id>\n"
            f"  <status>{task.status}</status>\n"
            f"  <command>{task.command}</command>\n"
            f"  <output_path>{task.output_path}</output_path>\n"
            f"  <summary>{output[:500]}</summary>\n"
            "</task_notification>"
        )
    return notifications


def _next_task_id() -> str:
    global _counter
    with _lock:
        _counter += 1
        return f"bg_{_counter:04d}"


def _output_path(repo_root: Path, task_id: str) -> Path:
    return repo_root / ".osc_agent" / "background" / f"{task_id}.out"


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
