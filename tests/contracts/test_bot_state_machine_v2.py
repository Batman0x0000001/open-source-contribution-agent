from __future__ import annotations

import asyncio

import pytest

from osc_agent.bot.outbox import DispatcherHealth, OutboxDispatcher
from osc_agent.bot.state_machine import BotJobStateMachine


@pytest.mark.parametrize(
    ("source", "event", "target"),
    [
        ("queued_plan", "claim_plan", "running_plan"),
        ("running_plan", "plan_ready", "waiting_approval"),
        ("running_plan", "plan_blocked", "blocked_plan"),
        ("blocked_plan", "reply_received", "queued_plan"),
        ("waiting_approval", "implementation_approved", "queued_implementation"),
        ("queued_implementation", "claim_implementation", "running_implementation"),
        ("running_implementation", "implementation_ready", "ready_to_publish"),
        ("ready_to_publish", "begin_publish", "publishing"),
        ("publishing", "publish_complete", "completed"),
    ],
)
def test_job_state_machine_transitions(source: str, event: str, target: str) -> None:
    assert BotJobStateMachine.transition(source, event) == target


def test_job_state_machine_rejects_invalid_transition() -> None:
    with pytest.raises(ValueError, match="BOT_INVALID_TRANSITION"):
        BotJobStateMachine.transition("queued_plan", "publish_complete")


def test_retry_requires_bound_phase_and_terminal_exhaustion() -> None:
    assert BotJobStateMachine.transition(
        "retry_wait", "resume_retry", retry_phase="implementation"
    ) == "queued_implementation"
    assert BotJobStateMachine.transition("retry_wait", "exhaust_retries") == "dead_letter"
    with pytest.raises(ValueError, match="RETRY_PHASE"):
        BotJobStateMachine.transition("retry_wait", "resume_retry")


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
