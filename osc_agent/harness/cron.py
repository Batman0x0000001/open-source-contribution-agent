"""
LLM 调用 schedule_check 创建 cron 任务
    ↓
cron 表达式校验（validate_cron）
    ↓
写入 scheduled_tasks.json（持久化）
    ↓
agent_loop 每轮开始
    ↓
collect_cron_notifications()
    ↓
读取所有 schedule
    ↓
cron_matches 判断当前时间是否命中
    ↓
命中 → 生成 <task_notification>
    ↓
写 last_fired_at（防重复触发）
    ↓
append_trace 记录触发事件
    ↓
注入到下一轮 LLM prompt
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from osc_agent.harness.trace import append_trace

_schedule_lock = threading.RLock()

CRON_TOOLS = [
    {
        "name": "schedule_check",
        "description": "Schedule a recurring reminder/check using a five-field cron expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cron": {
                    "type": "string",
                    "description": "Five-field cron expression, for example '*/5 * * * *' or '0 9 * * 1-5'.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Message to inject when the schedule fires.",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Whether the schedule is active immediately. Defaults to true.",
                },
            },
            "required": ["cron", "prompt"],
        },
    },
    {
        "name": "list_schedules",
        "description": "List persisted cron schedules for this repository.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel_schedule",
        "description": "Disable a cron schedule by id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "schedule_id": {
                    "type": "string",
                    "description": "Schedule id returned by schedule_check.",
                }
            },
            "required": ["schedule_id"],
        },
    },
]


@dataclass
class CronSchedule:
    id: str
    cron: str
    prompt: str
    enabled: bool
    created_at: str
    last_fired_at: str | None = None


def schedule_check(*, repo_root: Path, cron: str, prompt: str, enabled: bool = True) -> str:
    """创建持久化定时检查；写入前先校验 cron，避免坏表达式拖垮后续调度。"""
    error = validate_cron(cron)
    if error:
        return f"Error: {error}"
    if not prompt.strip():
        return "Error: prompt is required"

    with _schedule_lock:
        schedules = _load_schedules(repo_root)
        schedule = CronSchedule(
            id=f"cron_{uuid.uuid4().hex[:8]}",
            cron=cron.strip(),
            prompt=prompt.strip(),
            enabled=enabled,
            created_at=_now().isoformat(),
        )
        schedules.append(schedule)
        _save_schedules(repo_root, schedules)
    append_trace(repo_root, "schedule_check", {"schedule_id": schedule.id, "cron": schedule.cron})
    return json.dumps(asdict(schedule), ensure_ascii=False, indent=2)


def list_schedules(*, repo_root: Path) -> str:
    """返回当前 repo 的定时任务列表，包含已禁用任务，便于用户审计。"""
    with _schedule_lock:
        schedules = _load_schedules(repo_root)
    if not schedules:
        return "(no schedules)"
    return "\n".join(
        f"{schedule.id} [{'enabled' if schedule.enabled else 'disabled'}] {schedule.cron} -> {schedule.prompt}"
        for schedule in schedules
    )


def cancel_schedule(*, repo_root: Path, schedule_id: str) -> str:
    """取消任务采用禁用而非删除，保留审计痕迹并避免误删历史。"""
    with _schedule_lock:
        schedules = _load_schedules(repo_root)
        for schedule in schedules:
            if schedule.id == schedule_id:
                schedule.enabled = False
                _save_schedules(repo_root, schedules)
                append_trace(repo_root, "cancel_schedule", {"schedule_id": schedule_id})
                return f"Canceled schedule {schedule_id}"
        return f"Error: unknown schedule {schedule_id}"


def collect_cron_notifications(repo_root: Path, *, now: datetime | None = None) -> list[str]:
    """每轮 agent loop 开始时调用：检查到期任务并生成独立通知文本。"""
    current = now or _now()
    minute_marker = current.strftime("%Y-%m-%d %H:%M")
    with _schedule_lock:
        schedules = _load_schedules(repo_root)
        changed = False
        notifications: list[str] = []

        for schedule in schedules:
            if not schedule.enabled or schedule.last_fired_at == minute_marker:
                continue
            if not cron_matches(schedule.cron, current):
                continue
            # 同一分钟只触发一次；检查与标记在同一临界区，避免并发重复注入。
            schedule.last_fired_at = minute_marker
            changed = True
            notifications.append(
                "<task_notification>\n"
                f"  <task_id>{schedule.id}</task_id>\n"
                "  <status>scheduled</status>\n"
                f"  <cron>{schedule.cron}</cron>\n"
                f"  <summary>[Scheduled] {schedule.prompt}</summary>\n"
                "</task_notification>"
            )
            append_trace(repo_root, "cron_fired", {"schedule_id": schedule.id, "cron": schedule.cron})

        if changed:
            _save_schedules(repo_root, schedules)
        return notifications


def validate_cron(cron: str) -> str | None:
    fields = cron.strip().split()
    if len(fields) != 5:
        return "cron expression must have exactly five fields"
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    for field, (minimum, maximum) in zip(fields, ranges):
        error = _validate_field(field, minimum, maximum)
        if error:
            return error
    return None


def cron_matches(cron: str, dt: datetime) -> bool:
    """实现五字段 cron 匹配；DOM 和 DOW 同时受限时使用传统 OR 语义。"""
    if validate_cron(cron):
        return False
    minute, hour, dom, month, dow = cron.strip().split()
    cron_dow = (dt.weekday() + 1) % 7
    if not (
        _field_matches(minute, dt.minute)
        and _field_matches(hour, dt.hour)
        and _field_matches(month, dt.month)
    ):
        return False

    dom_ok = _field_matches(dom, dt.day)
    dow_ok = _field_matches(dow, cron_dow) or (cron_dow == 0 and _field_matches(dow, 7))
    if dom == "*" and dow == "*":
        return True
    if dom == "*":
        return dow_ok
    if dow == "*":
        return dom_ok
    return dom_ok or dow_ok


def schedules_path(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "scheduled_tasks.json"


def _load_schedules(repo_root: Path) -> list[CronSchedule]:
    with _schedule_lock:
        path = schedules_path(repo_root)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        schedules: list[CronSchedule] = []
        for item in data.get("schedules", []):
            cron = str(item.get("cron", ""))
            if validate_cron(cron):
                continue
            schedules.append(
                CronSchedule(
                    id=str(item.get("id", "")),
                    cron=cron,
                    prompt=str(item.get("prompt", "")),
                    enabled=bool(item.get("enabled", True)),
                    created_at=str(item.get("created_at", "")),
                    last_fired_at=item.get("last_fired_at"),
                )
            )
        return schedules


def _save_schedules(repo_root: Path, schedules: list[CronSchedule]) -> None:
    with _schedule_lock:
        path = schedules_path(repo_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schedules": [asdict(schedule) for schedule in schedules]}
        temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()


def _validate_field(field: str, minimum: int, maximum: int) -> str | None:
    if field == "*":
        return None
    for part in field.split(","):
        base, _, step = part.partition("/")
        if step and (not step.isdigit() or int(step) <= 0):
            return f"invalid cron step: {part}"
        if base == "*":
            continue
        if "-" in base:
            start_text, end_text = base.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                return f"invalid cron range: {part}"
            start, end = int(start_text), int(end_text)
            if start > end or start < minimum or end > maximum:
                return f"cron value out of range: {part}"
            continue
        if not base.isdigit():
            return f"invalid cron value: {part}"
        value = int(base)
        if value < minimum or value > maximum:
            return f"cron value out of range: {part}"
    return None


def _field_matches(field: str, value: int) -> bool:
    for part in field.split(","):
        base, _, step_text = part.partition("/")
        step = int(step_text) if step_text else 1
        if base == "*":
            if value % step == 0:
                return True
            continue
        if "-" in base:
            start, end = [int(piece) for piece in base.split("-", 1)]
            if start <= value <= end and (value - start) % step == 0:
                return True
            continue
        if int(base) == value:
            return True
    return False


def _now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Shanghai"))
