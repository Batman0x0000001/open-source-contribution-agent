"""定义 Bot 作业生命周期及其合法状态转换。"""

from __future__ import annotations

from enum import Enum

from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed


class JobEvent(str, Enum):
    CLAIM_PLAN = "claim_plan"
    PLAN_READY = "plan_ready"
    PLAN_BLOCKED = "plan_blocked"
    REPLY_RECEIVED = "reply_received"
    IMPLEMENTATION_APPROVED = "implementation_approved"
    CLAIM_IMPLEMENTATION = "claim_implementation"
    IMPLEMENTATION_READY = "implementation_ready"
    BEGIN_PUBLISH = "begin_publish"
    PUBLISH_COMPLETE = "publish_complete"
    SCHEDULE_RETRY = "schedule_retry"
    RESUME_RETRY = "resume_retry"
    EXHAUST_RETRIES = "exhaust_retries"
    MARK_STALE = "mark_stale"
    CANCEL = "cancel"


class BotJobStateMachine(StateMachine):
    queued_plan = State(initial=True)
    running_plan = State()
    blocked_plan = State()
    waiting_approval = State()
    queued_implementation = State()
    running_implementation = State()
    ready_to_publish = State()
    publishing = State()
    retry_wait = State()
    completed = State(final=True)
    stale = State(final=True)
    dead_letter = State(final=True)
    cancelled = State(final=True)

    claim_plan = queued_plan.to(running_plan)
    plan_ready = running_plan.to(waiting_approval)
    plan_blocked = running_plan.to(blocked_plan)
    reply_received = blocked_plan.to(queued_plan)
    implementation_approved = waiting_approval.to(queued_implementation)
    claim_implementation = queued_implementation.to(running_implementation)
    implementation_ready = running_implementation.to(ready_to_publish)
    begin_publish = ready_to_publish.to(publishing)
    publish_complete = publishing.to(completed)
    schedule_retry = (
        running_plan.to(retry_wait)
        | running_implementation.to(retry_wait)
        | publishing.to(retry_wait)
    )
    resume_plan = retry_wait.to(queued_plan)
    resume_implementation = retry_wait.to(queued_implementation)
    resume_publish = retry_wait.to(ready_to_publish)
    exhaust_retries = retry_wait.to(dead_letter)
    mark_stale = (
        waiting_approval.to(stale)
        | running_implementation.to(stale)
        | ready_to_publish.to(stale)
        | publishing.to(stale)
    )
    cancel = (
        queued_plan.to(cancelled)
        | running_plan.to(cancelled)
        | blocked_plan.to(cancelled)
        | waiting_approval.to(cancelled)
        | queued_implementation.to(cancelled)
        | running_implementation.to(cancelled)
        | ready_to_publish.to(cancelled)
        | retry_wait.to(cancelled)
    )

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        return any(state.final and state.value == status for state in cls.states)

    @classmethod
    def transition(cls, current: str, event: JobEvent, *, retry_phase: str | None = None) -> str:
        machine = cls(start_value=current)
        trigger = event.value
        if event is JobEvent.RESUME_RETRY:
            if retry_phase not in {"plan", "implementation", "publish"}:
                raise ValueError("BOT_RETRY_PHASE_REQUIRED")
            trigger = f"resume_{retry_phase}"
        try:
            getattr(machine, trigger)()
        except (AttributeError, ValueError, TransitionNotAllowed) as exc:
            raise ValueError(f"BOT_INVALID_TRANSITION:{current}:{event.value}") from exc
        return str(next(iter(machine.configuration)).value)
