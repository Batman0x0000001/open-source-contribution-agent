"""验证Bot Worker的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.persistence.store import BotStore
from osc_agent.bot.worker.agent_jobs import (
    BotJobLeaseLost,
    BotModelContractMismatch,
    ImplementationJobExecutor,
    consume_runtime_events,
)
from osc_agent.bot.worker.coordinator import BotWorker
from osc_agent.runtime.events import (
    AssistantDelta,
    Complete,
    ModelRequestStarted,
    RunCompleted,
)


IMAGE_ID = "sha256:" + "a" * 64


def _job(*, status: str = "running_implementation") -> BotJob:
    return BotJob(
        job_id=str(uuid4()),
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=1,
        issue_number=1,
        issue_url="https://github.com/owner/repo/issues/1",
        base_sha="b" * 40,
        image_id=IMAGE_ID,
        status=status,
    )


def test_runtime_progress_is_redacted_and_throttled() -> None:
    job = _job(status="running_plan").model_copy(update={"lease_owner": "worker"})

    class Store:
        progress: list[str] = []

        def get_job(self, _job_id):
            return job

        def record_progress(self, _job_id, _worker_id, event_type):
            self.progress.append(event_type)
            return job

        def heartbeat(self, _job_id, _worker_id):
            return job

    async def events():
        yield ModelRequestStarted(session_id="session", round_number=1)
        yield AssistantDelta(text="sensitive model output must not be persisted")
        yield RunCompleted(transition=Complete(reason="done"))

    store = Store()
    sessions = SimpleNamespace(renew_lease=lambda _session_id: None)
    completed = asyncio.run(
        consume_runtime_events(
            events(), store=store, sessions=sessions, session_id="session",
            worker_id="worker", job_id=job.job_id, phase="plan"
        )
    )

    assert completed is True
    assert store.progress == ["plan:model_request_started", "plan:run_completed"]
    assert "sensitive" not in " ".join(store.progress)


def test_repository_policy_violation_immediately_terminates_job(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    job = _job().model_copy(update={"lease_owner": "worker"})
    store.create_job(job)
    executor = object.__new__(ImplementationJobExecutor)
    executor.store = store  # type: ignore[assignment]
    executor.bot_settings = SimpleNamespace(worker_id="worker")  # type: ignore[assignment]

    executor._terminate_repository_policy(job.job_id, "protected path")

    terminated = store.get_job(job.job_id)
    assert terminated is not None
    assert terminated.status == "cancelled"
    assert terminated.error_code == "REPOSITORY_POLICY_VIOLATION"
    assert terminated.error_message == "protected path"


def test_model_contract_mismatch_dead_letters_without_running_agent() -> None:
    job = _job(status="running_plan").model_copy(update={"lease_owner": "worker"})
    transitions: list[tuple[str, dict[str, object]]] = []

    class Store:
        def claim_job(self, _worker_id, phase=None):
            return job

        def get_job(self, _job_id):
            return job

        def apply_job_event_with_outbox(self, **changes):
            transitions.append(("retry_wait", changes))
            return job.model_copy(update={"status": "retry_wait", "version": job.version + 1})

        def apply_job_event(self, **changes):
            transitions.append(("dead_letter", changes))
            return job.model_copy(update={"status": "dead_letter", "version": job.version + 2})

    worker = object.__new__(BotWorker)
    worker.store = Store()  # type: ignore[assignment]
    worker.bot_settings = SimpleNamespace(worker_id="worker")  # type: ignore[assignment]

    async def reject(_job):
        raise BotModelContractMismatch("model mismatch")

    worker.plan_executor = SimpleNamespace(execute=reject)  # type: ignore[assignment]

    assert asyncio.run(worker.run_once("plan")) is True
    assert [status for status, _ in transitions] == ["retry_wait", "dead_letter"]
    assert transitions[0][1]["error_code"] == "BOT_MODEL_CONTRACT_MISMATCH"


def test_reclaimed_worker_does_not_schedule_retry_or_complete() -> None:
    claimed = _job(status="running_plan").model_copy(update={"lease_owner": "worker-1"})
    reclaimed = claimed.model_copy(
        update={"lease_owner": "worker-2", "version": claimed.version + 1}
    )
    transitions: list[dict[str, object]] = []

    class Store:
        def claim_job(self, _worker_id, phase=None):
            return claimed

        def get_job(self, _job_id):
            return reclaimed

        def apply_job_event_with_outbox(self, **changes):
            transitions.append(changes)

    worker = object.__new__(BotWorker)
    worker.store = Store()  # type: ignore[assignment]
    worker.bot_settings = SimpleNamespace(worker_id="worker-1")  # type: ignore[assignment]

    async def fail(_job):
        raise RuntimeError("old worker resumed after lease expiry")

    worker.plan_executor = SimpleNamespace(execute=fail)  # type: ignore[assignment]

    assert asyncio.run(worker.run_once("plan")) is True
    assert transitions == []


def test_runtime_stream_stops_when_job_lease_is_reclaimed() -> None:
    job = _job(status="running_plan").model_copy(update={"lease_owner": "worker-2"})

    class Store:
        def get_job(self, _job_id):
            return job

    async def events():
        yield ModelRequestStarted(session_id="session", round_number=1)

    with pytest.raises(BotJobLeaseLost, match="reclaimed"):
        asyncio.run(
            consume_runtime_events(
                events(),
                store=Store(),  # type: ignore[arg-type]
                sessions=SimpleNamespace(renew_lease=lambda _session_id: None),
                session_id="session",
                worker_id="worker-1",
                job_id=job.job_id,
                phase="plan",
            )
        )


def test_worker_uses_independent_phase_slots_and_graceful_shutdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import osc_agent.bot.worker.service as server_module

    started = asyncio.Event()
    slot_names: set[str] = set()

    class Store:
        def __init__(self, _path):
            pass

        def initialize(self):
            pass

    class Worker:
        def __init__(self, **_kwargs):
            pass

        async def run_once(self, _phase):
            task = asyncio.current_task()
            assert task is not None
            slot_names.add(task.get_name())
            if len(slot_names) == 3:
                started.set()
            return False

    monkeypatch.setattr(server_module, "BotStore", Store)
    monkeypatch.setattr(server_module, "BotWorker", Worker)
    monkeypatch.setattr(server_module, "load_repository_catalog", lambda _path: object())
    monkeypatch.setattr(server_module, "load_agent_settings", lambda: object())
    settings = SimpleNamespace(
        database_path=tmp_path / "bot.sqlite3",
        repositories_config=tmp_path / "repositories.yml",
        max_concurrent_plans=2,
        max_concurrent_implementations=1,
        shutdown_timeout_seconds=1,
    )

    async def scenario() -> None:
        shutdown = asyncio.Event()
        running = asyncio.create_task(server_module.run_worker_forever(settings, shutdown))
        await asyncio.wait_for(started.wait(), timeout=2)
        shutdown.set()
        await asyncio.wait_for(running, timeout=2)

    asyncio.run(scenario())
    assert len({name for name in slot_names if name.startswith("worker-plan-")}) == 2
    assert len({name for name in slot_names if name.startswith("worker-implementation-")}) == 1
