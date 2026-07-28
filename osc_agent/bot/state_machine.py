from __future__ import annotations

from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed


TRANSITIONS: tuple[tuple[str, str, str], ...] = (
    ("queued_plan", "claim_plan", "running_plan"),
    ("running_plan", "plan_ready", "waiting_approval"),
    ("running_plan", "plan_blocked", "blocked_plan"),
    ("blocked_plan", "reply_received", "queued_plan"),
    ("waiting_approval", "implementation_approved", "queued_implementation"),
    ("queued_implementation", "claim_implementation", "running_implementation"),
    ("running_implementation", "implementation_ready", "ready_to_publish"),
    ("ready_to_publish", "begin_publish", "publishing"),
    ("publishing", "publish_complete", "completed"),
    ("running_plan", "schedule_retry", "retry_wait"),
    ("running_implementation", "schedule_retry", "retry_wait"),
    ("publishing", "schedule_retry", "retry_wait"),
    ("retry_wait", "resume_plan", "queued_plan"),
    ("retry_wait", "resume_implementation", "queued_implementation"),
    ("retry_wait", "resume_publish", "ready_to_publish"),
    ("retry_wait", "exhaust_retries", "dead_letter"),
    ("waiting_approval", "mark_stale", "stale"),
    ("running_implementation", "mark_stale", "stale"),
    ("ready_to_publish", "mark_stale", "stale"),
    ("publishing", "mark_stale", "stale"),
    ("queued_plan", "cancel", "cancelled"),
    ("running_plan", "cancel", "cancelled"),
    ("blocked_plan", "cancel", "cancelled"),
    ("waiting_approval", "cancel", "cancelled"),
    ("queued_implementation", "cancel", "cancelled"),
    ("running_implementation", "cancel", "cancelled"),
    ("ready_to_publish", "cancel", "cancelled"),
    ("publishing", "cancel", "cancelled"),
    ("retry_wait", "cancel", "cancelled"),
)


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
        | publishing.to(cancelled)
        | retry_wait.to(cancelled)
    )

    @classmethod
    def transition(cls, current: str, event: str, *, retry_phase: str | None = None) -> str:
        machine = cls(start_value=current)
        trigger = event
        if event == "resume_retry":
            if retry_phase not in {"plan", "implementation", "publish"}:
                raise ValueError("BOT_RETRY_PHASE_REQUIRED")
            trigger = f"resume_{retry_phase}"
        try:
            getattr(machine, trigger)()
        except (AttributeError, ValueError, TransitionNotAllowed) as exc:
            raise ValueError(f"BOT_INVALID_TRANSITION:{current}:{event}") from exc
        return str(next(iter(machine.configuration)).value)

    @classmethod
    def mermaid(cls) -> str:
        lines = ["stateDiagram-v2", "    [*] --> queued_plan"]
        lines.extend(f"    {source} --> {target}: {event}" for source, event, target in TRANSITIONS)
        lines.extend(
            f"    {state} --> [*]"
            for state in ("completed", "stale", "dead_letter", "cancelled")
        )
        return "\n".join(lines)

    @classmethod
    def transition_table(cls) -> str:
        rows = ["| From | Event | To |", "|---|---|---|"]
        rows.extend(f"| `{source}` | `{event}` | `{target}` |" for source, event, target in TRANSITIONS)
        return "\n".join(rows)
