from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    prohibited = {"secret", "token", "prompt", "issue_body", "tool_content", "file_content"}
    safe = {
        key: value for key, value in fields.items()
        if value is not None and not any(marker in key.lower() for marker in prohibited)
    }
    safe.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "service": safe.get("service", "unknown"),
        "event": event,
    })
    logger.info(json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str))


try:
    from prometheus_client import Counter, Gauge, Histogram

    OUTBOX_ITERATIONS = Counter("osa_outbox_iterations_total", "Outbox iterations", ["outcome"])
    OUTBOX_UP = Gauge("osa_outbox_dispatcher_up", "Whether the dispatcher is running")
    OUTBOX_HEARTBEAT = Gauge("osa_outbox_dispatcher_heartbeat_age_seconds", "Dispatcher heartbeat age")
    JOB_RUNS = Counter("osa_jobs_total", "Bot jobs", ["phase", "outcome"])
    JOB_ACTIVE = Gauge("osa_jobs_active", "Active jobs", ["state"])
    JOB_QUEUED_AT = Gauge("osa_job_queued_timestamp_seconds", "Oldest queued job timestamp", ["phase"])
    JOB_DURATION = Histogram("osa_job_duration_seconds", "Bot job duration", ["phase"])
    JOB_RETRIES = Counter("osa_job_retries_total", "Bot job retries", ["phase"])
    OUTBOX_FAILURES = Counter("osa_outbox_handler_failures_total", "Outbox handler failures", ["kind"])
    OUTBOX_CRASHES = Counter("osa_outbox_task_crashes_total", "Outbox task crashes")
    OUTBOX_DEPTH = Gauge("osa_outbox_depth", "Pending outbox events")
    OUTBOX_OLDEST = Gauge("osa_outbox_oldest_seconds", "Age of oldest outbox event")
    DEAD_LETTERS = Gauge("osa_dead_letters", "Dead-letter jobs and events")
    AGENT_RUNS = Counter("osa_agent_runs_total", "Agent runs", ["profile", "outcome"])
    AGENT_DURATION = Histogram("osa_agent_run_duration_seconds", "Agent run duration", ["profile"])
    TOOL_CALLS = Counter("osa_tool_calls_total", "Tool calls", ["tool", "outcome"])
    WORKER_ACTIVE = Gauge("osa_worker_active_jobs", "Worker active jobs", ["phase"])
    WORKER_HEARTBEAT = Gauge("osa_worker_heartbeat_age_seconds", "Worker heartbeat age")
    PUBLISH_TOTAL = Counter("osa_publish_total", "Publish attempts", ["outcome", "mode"])
except ImportError:  # Optional outside the bot installation extra.
    OUTBOX_ITERATIONS = OUTBOX_UP = OUTBOX_HEARTBEAT = JOB_RUNS = JOB_ACTIVE = JOB_QUEUED_AT = JOB_DURATION = None
    JOB_RETRIES = OUTBOX_FAILURES = OUTBOX_CRASHES = OUTBOX_DEPTH = OUTBOX_OLDEST = None
    DEAD_LETTERS = AGENT_RUNS = AGENT_DURATION = TOOL_CALLS = WORKER_ACTIVE = None
    WORKER_HEARTBEAT = PUBLISH_TOTAL = None
