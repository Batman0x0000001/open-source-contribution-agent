"""验证Bot 作业状态机的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio

import pytest

from osc_agent.bot.control.outbox import DispatcherHealth, OutboxDispatcher
from osc_agent.bot.domain.state_machine import BotJobStateMachine, JobEvent


@pytest.mark.parametrize(
    ("source", "event", "target"),
    [
        ("queued_plan", JobEvent.CLAIM_PLAN, "running_plan"),
        ("running_plan", JobEvent.PLAN_READY, "waiting_approval"),
        ("running_plan", JobEvent.PLAN_BLOCKED, "blocked_plan"),
        ("blocked_plan", JobEvent.REPLY_RECEIVED, "queued_plan"),
        ("waiting_approval", JobEvent.IMPLEMENTATION_APPROVED, "queued_implementation"),
        ("queued_implementation", JobEvent.CLAIM_IMPLEMENTATION, "running_implementation"),
        ("running_implementation", JobEvent.IMPLEMENTATION_READY, "ready_to_publish"),
        ("ready_to_publish", JobEvent.BEGIN_PUBLISH, "publishing"),
        ("publishing", JobEvent.PUBLISH_COMPLETE, "completed"),
        ("running_plan", JobEvent.SCHEDULE_RETRY, "retry_wait"),
        ("retry_wait", JobEvent.EXHAUST_RETRIES, "dead_letter"),
        ("waiting_approval", JobEvent.MARK_STALE, "stale"),
        ("queued_plan", JobEvent.CANCEL, "cancelled"),
    ],
)
def test_job_state_machine_transitions(source: str, event: JobEvent, target: str) -> None:
    assert BotJobStateMachine.transition(source, event) == target


def test_job_state_machine_rejects_invalid_transition() -> None:
    with pytest.raises(ValueError, match="BOT_INVALID_TRANSITION"):
        BotJobStateMachine.transition("queued_plan", JobEvent.PUBLISH_COMPLETE)


def test_retry_requires_bound_phase_and_terminal_exhaustion() -> None:
    assert BotJobStateMachine.transition(
        "retry_wait", JobEvent.RESUME_RETRY, retry_phase="implementation"
    ) == "queued_implementation"
    assert BotJobStateMachine.transition(
        "retry_wait", JobEvent.EXHAUST_RETRIES
    ) == "dead_letter"
    with pytest.raises(ValueError, match="RETRY_PHASE"):
        BotJobStateMachine.transition("retry_wait", JobEvent.RESUME_RETRY)


def test_dispatcher_propagates_processor_failure() -> None:
    class Broken:
        async def run_once(self) -> bool:
            raise RuntimeError("database unavailable")

    health = DispatcherHealth()
    dispatcher = OutboxDispatcher(Broken(), health)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="database unavailable"):
        asyncio.run(dispatcher.run_forever(asyncio.Event()))
    assert health.running is False
    assert health.last_error == "database unavailable"
