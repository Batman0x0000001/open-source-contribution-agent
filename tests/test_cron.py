from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from osc_agent.agent_loop import TOOLS, build_tool_handlers
from osc_agent.harness.cron import (
    cancel_schedule,
    collect_cron_notifications,
    cron_matches,
    list_schedules,
    schedule_check,
    schedules_path,
    validate_cron,
)


def _schedule_id(output: str) -> str:
    return json.loads(output)["id"]


def test_validate_cron_rejects_invalid_expression():
    assert validate_cron("* * * *") == "cron expression must have exactly five fields"
    assert "out of range" in str(validate_cron("99 * * * *"))
    assert "invalid cron step" in str(validate_cron("*/0 * * * *"))


def test_cron_matches_basic_patterns():
    dt = datetime(2026, 7, 6, 9, 10, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert cron_matches("*/5 9 * * 1-5", dt)
    assert cron_matches("10 9 1 * 1", dt)
    assert not cron_matches("11 9 * * *", dt)


def test_schedule_check_persists_and_can_be_listed(tmp_path):
    output = schedule_check(repo_root=tmp_path, cron="*/5 * * * *", prompt="check tests")
    schedule_id = _schedule_id(output)

    assert schedules_path(tmp_path).exists()
    assert schedule_id in list_schedules(repo_root=tmp_path)
    assert "*/5 * * * * -> check tests" in list_schedules(repo_root=tmp_path)


def test_cancel_schedule_disables_job(tmp_path):
    schedule_id = _schedule_id(schedule_check(repo_root=tmp_path, cron="* * * * *", prompt="scan TODO"))

    result = cancel_schedule(repo_root=tmp_path, schedule_id=schedule_id)
    listing = list_schedules(repo_root=tmp_path)

    assert result == f"Canceled schedule {schedule_id}"
    assert f"{schedule_id} [disabled]" in listing


def test_collect_cron_notifications_fires_once_per_minute(tmp_path):
    now = datetime(2026, 7, 4, 12, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    schedule_id = _schedule_id(schedule_check(repo_root=tmp_path, cron="30 12 * * *", prompt="rerun tests"))

    notifications = collect_cron_notifications(tmp_path, now=now)
    repeated = collect_cron_notifications(tmp_path, now=now)

    assert len(notifications) == 1
    assert f"<task_id>{schedule_id}</task_id>" in notifications[0]
    assert "[Scheduled] rerun tests" in notifications[0]
    assert repeated == []


def test_agent_loop_registers_cron_tools(tmp_path):
    tool_names = {tool["name"] for tool in TOOLS}
    handlers = build_tool_handlers(tmp_path)

    assert {"schedule_check", "list_schedules", "cancel_schedule"} <= tool_names
    output = handlers["schedule_check"](cron="0 9 * * *", prompt="check background tasks")
    schedule_id = _schedule_id(output)
    assert schedule_id in handlers["list_schedules"]()
    assert handlers["cancel_schedule"](schedule_id=schedule_id) == f"Canceled schedule {schedule_id}"
