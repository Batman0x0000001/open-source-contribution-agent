# Bot observability runbook

本文适用于 Open Source Contribution Agent `0.3.2`。

Control and Worker emit structured JSON. The stable envelope is `timestamp`, `level`, `service`,
`event`, `job_id`, `session_id`, `repository`, `issue_number`, state transition, phase, attempt,
duration, error code, GitHub delivery ID and execution-contract hash. Logs must not contain secrets,
full prompts, Issue bodies, Tool inputs/outputs or repository file contents.
Job 的 `last_progress_event` 只包含 phase 和 RuntimeEvent 类型，并最多每五秒更新一次（终态
事件例外）；`last_progress_at` 可用于识别仍在运行但没有新事件的 Agent。

Prometheus exports Job counts/active/duration/retries, dispatcher up/heartbeat/iterations/failures/
crashes/depth/oldest age, dead letters, Agent runs/duration, Tool calls, Worker active/heartbeat and
publish outcomes. The deploy alert file covers dispatcher or target down, stale heartbeat, dead
letters, ten-minute Outbox age, fifteen-minute queued jobs, two-minute Worker heartbeat and a
15-minute failure rate over 30%.

Triage order:

1. Check `/health/ready`; 503 identifies schema/database or dispatcher health failure.
2. Check `osc_outbox_dispatcher_up`, heartbeat age and `osc_outbox_oldest_seconds`.
3. Inspect JSON events by `job_id` and `execution_contract_hash`; never paste raw prompts into logs.
4. Use `/osc-agent status` for active state. Use `/osc-agent retry` only in `retry_wait`.
5. A dead letter is not user-retryable. An administrator must inspect the audit trail and use the
   maintenance CLI in a stopped-service window.

An unclassified Dispatcher exception is a process failure by design: readiness drops, Uvicorn is
cancelled, the original exception reaches the process top level, and systemd restarts Control.
Retryable and terminal handler errors remain contained; poison events become dead letters after the
bounded attempt limit.
