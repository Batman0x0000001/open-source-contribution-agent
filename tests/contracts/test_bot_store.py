"""验证Bot SQLite Store的契约、边界条件与回归行为。"""

from __future__ import annotations

from osc_agent.runtime.state import CapabilityScope

import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest

from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.artifacts import IssuePlanArtifact
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.state_machine import JobEvent
from osc_agent.bot.persistence import session_store as session_store_module
from osc_agent.bot.persistence.schema import SCHEMA_EPOCH, STATE_MODEL_REVISION
from osc_agent.bot.persistence.session_store import SqliteSessionStore
from osc_agent.bot.persistence.store import BotStore
from osc_agent.runtime.messages import RuntimeMessage, TextBlock
from osc_agent.runtime.session import SessionMetadata
from tests.runtime_factories import agent_run_state



def _job() -> BotJob:
    return BotJob(
        job_id=str(uuid4()),
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=2,
        issue_number=3,
        issue_url="https://github.com/owner/repo/issues/3",
        base_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        status="queued_plan",
    )


def test_initialize_makes_a_new_database_group_writable(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "bot.sqlite3"
    chmod_calls: list[tuple[Path, int]] = []
    real_chmod = Path.chmod

    def record_chmod(target: Path, mode: int, *, follow_symlinks: bool = True) -> None:
        chmod_calls.append((target, mode))
        real_chmod(target, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "chmod", record_chmod)

    BotStore(path).initialize()

    assert (path.resolve(), 0o660) in chmod_calls


def test_initialize_rejects_the_previous_bot_state_epoch(tmp_path: Path) -> None:
    path = tmp_path / "bot.sqlite3"
    store = BotStore(path)
    store.initialize()
    assert (SCHEMA_EPOCH, STATE_MODEL_REVISION) == (4, "bot-job-v3-session-v6")
    with store.connect() as connection:
        connection.execute(
            "UPDATE schema_meta SET schema_epoch=3, state_model_revision='bot-job-v3-session-v5'"
        )

    with pytest.raises(ValueError, match="BOT_SCHEMA_EPOCH_MISMATCH"):
        store.initialize()


def test_sqlite_job_delivery_version_and_claim(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    assert store.register_delivery("delivery-1", "issue_comment") is True
    assert store.register_delivery("delivery-1", "issue_comment") is False
    job = _job()
    store.create_job(job)
    with pytest.raises(ValueError, match="VERSION_CONFLICT"):
        store.update_job_fields(job.job_id, expected_version=99, error_code="failed")
    for field, value in (
        ("status", "completed"),
        ("version", 99),
        ("updated_at", "never"),
    ):
        with pytest.raises(ValueError, match="RESERVED_FIELDS"):
            store.update_job_fields(job.job_id, expected_version=job.version, **{field: value})
    with pytest.raises(ValueError, match="RESERVED_FIELDS"):
        store.apply_job_event(
            job_id=job.job_id,
            expected_version=job.version,
            event=JobEvent.CLAIM_PLAN,
            status="running_plan",
        )
    assert store.claim_job("worker") is None  # Workspace 尚未由可信 Control 准备。
    prepared = store.update_job_fields(
        job.job_id,
        expected_version=job.version,
        plan_workspace_path=str(tmp_path / "workspace"),
        plan_workspace_ready=True,
    )
    assert store.claim_job("worker", phase="implementation") is None
    claimed = store.claim_job("worker", phase="plan")
    assert claimed is not None
    assert claimed.status == "running_plan"
    assert claimed.lease_owner == "worker"
    assert claimed.version == prepared.version + 1

    expired = store.update_job_fields(
        job.job_id,
        expected_version=claimed.version,
        lease_until="2000-01-01T00:00:00+00:00",
    )
    reclaimed = store.claim_job("worker-2", phase="plan")
    assert reclaimed is not None
    assert reclaimed.status == "running_plan"
    assert reclaimed.plan_attempts == expired.plan_attempts + 1
    assert reclaimed.lease_owner == "worker-2"

    progressed = store.record_progress(
        job.job_id, "worker-2", "plan:model_request_started"
    )
    assert progressed.last_progress_event == "plan:model_request_started"
    assert progressed.last_progress_at is not None


def test_active_job_unique_index_uses_stable_repository_id(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    first = _job()
    store.create_job(first)
    duplicate = _job().model_copy(update={
        "repository_full_name": "renamed/repo",
        "issue_number": first.issue_number,
        "repository_id": first.repository_id,
    })
    with pytest.raises(sqlite3.IntegrityError):
        store.create_job(duplicate)


def test_sqlite_session_store_replays_strict_event_chain(tmp_path: Path) -> None:
    database = BotStore(tmp_path / "bot.sqlite3")
    database.initialize()
    sessions = SqliteSessionStore(database)
    session_id = str(uuid4())
    metadata = SessionMetadata(
        schema_version=6,
        session_id=session_id,
        workspace_root=str(tmp_path),
        model="model",
    )
    sessions.create(metadata, agent_run_state(str(tmp_path), capabilities=CapabilityScope(allowed_tools=frozenset({"read_file"}))))
    sessions.append_message(
        session_id,
        RuntimeMessage(role="user", content=[TextBlock(text="goal")]),
    )
    sessions.save_state(session_id, agent_run_state(str(tmp_path), status="completed"))
    snapshot = sessions.load(session_id)
    assert snapshot is not None
    assert snapshot.messages[0].content[0].text == "goal"
    assert snapshot.state.last_status == "completed"
    with sessions.lease(session_id):
        with pytest.raises(ValueError, match="SESSION_IN_USE"):
            with sessions.lease(session_id):
                pass


def test_sqlite_session_lease_is_short_and_renewable(tmp_path: Path) -> None:
    database = BotStore(tmp_path / "bot.sqlite3")
    database.initialize()
    sessions = SqliteSessionStore(database)
    assert sessions.lease_seconds == 60

    with sessions.lease("session"):
        with database.connect() as connection:
            connection.execute(
                "UPDATE session_leases SET lease_until='2000-01-01T00:00:00+00:00' "
                "WHERE session_id='session'"
            )
        sessions.renew_lease("session")
        with database.connect() as connection:
            renewed = connection.execute(
                "SELECT lease_until FROM session_leases WHERE session_id='session'"
            ).fetchone()[0]
        assert renewed > "2000-01-01T00:00:00+00:00"

    with pytest.raises(ValueError, match="SESSION_LEASE_LOST"):
        sessions.renew_lease("session")


def test_reclaimed_worker_cannot_write_job_or_artifact(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    job = _job()
    store.create_job(job)
    prepared = store.update_job_fields(
        job.job_id,
        expected_version=job.version,
        plan_workspace_path=str(tmp_path / "workspace"),
        plan_workspace_ready=True,
    )
    claimed = store.claim_job("worker-1", phase="plan")
    assert claimed is not None and claimed.version == prepared.version + 1
    expired = store.update_job_fields(
        job.job_id,
        expected_version=claimed.version,
        required_lease_owner="worker-1",
        lease_until="2000-01-01T00:00:00+00:00",
    )
    reclaimed = store.claim_job("worker-2", phase="plan")
    assert reclaimed is not None and reclaimed.version == expired.version + 1
    plan = IssuePlanArtifact(
        status="ready",
        base_sha=job.base_sha,
        execution_contract_hash=job.execution_contract_hash,
        summary="ready",
        plan_markdown="plan",
    )

    with pytest.raises(ValueError, match="BOT_JOB_LEASE_LOST"):
        store.apply_job_event(
            job_id=job.job_id,
            expected_version=reclaimed.version,
            event=JobEvent.PLAN_READY,
            required_lease_owner="worker-1",
        )
    with pytest.raises(ValueError, match="BOT_JOB_LEASE_LOST"):
        store.save_artifact(
            job.job_id,
            plan,
            required_lease_owner="worker-1",
        )
    assert store.latest_plan_artifact(job.job_id) is None


def test_sqlite_session_creation_rolls_back_both_initial_records(
    monkeypatch, tmp_path: Path
) -> None:
    database = BotStore(tmp_path / "bot.sqlite3")
    database.initialize()
    sessions = SqliteSessionStore(database)
    session_id = str(uuid4())
    real_append = session_store_module.append_session_record
    calls = 0

    def fail_on_initial_state(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated state write failure")
        real_append(*args, **kwargs)

    monkeypatch.setattr(session_store_module, "append_session_record", fail_on_initial_state)
    with pytest.raises(RuntimeError, match="simulated state write failure"):
        sessions.create(
            SessionMetadata(
                schema_version=6,
                session_id=session_id,
                workspace_root=str(tmp_path),
                model="model",
            ),
            agent_run_state(str(tmp_path)),
        )

    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM session_records WHERE session_id=?", (session_id,)
        ).fetchone()[0]
    assert count == 0


def test_sqlite_session_overview_uses_latest_record_timestamp(tmp_path: Path) -> None:
    database = BotStore(tmp_path / "bot.sqlite3")
    database.initialize()
    sessions = SqliteSessionStore(database)
    session_id = str(uuid4())
    sessions.create(
        SessionMetadata(
            schema_version=6,
            session_id=session_id,
            workspace_root=str(tmp_path),
            model="model",
        ),
        agent_run_state(str(tmp_path)),
    )
    with database.connect() as connection:
        expected = connection.execute(
            "SELECT MAX(created_at) FROM session_records WHERE session_id=?", (session_id,)
        ).fetchone()[0]

    overview = sessions.list_overviews()[0]
    assert overview.session_id == session_id
    assert overview.updated_at == expected


def test_outbox_claim_is_leased_and_recoverable(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    job = _job()
    store.create_job(job)
    assert store.enqueue_outbox(
        OutboxEvent(
            event_id="event-1",
            job_id=job.job_id,
            kind="prepare",
            idempotency_key="prepare:one",
            payload={"phase": "plan"},
        )
    )
    claimed = store.next_outbox(lease_seconds=60)
    assert claimed is not None
    assert claimed.status == "processing" and claimed.attempts == 1
    assert store.next_outbox() is None
    store.fail_outbox(claimed.event_id, "retry", delay_seconds=0)
    retried = store.next_outbox()
    assert retried is not None and retried.attempts == 2
    store.complete_outbox(retried.event_id)
    assert store.next_outbox() is None


def test_unknown_persisted_outbox_kind_can_be_claimed_for_dispatch(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    job = _job()
    store.create_job(job)
    assert store.enqueue_outbox(
        OutboxEvent(
            event_id="future-event",
            job_id=job.job_id,
            kind="future_delivery",
            idempotency_key="future:one",
            payload={},
        )
    )

    claimed = store.next_outbox()
    assert claimed is not None and claimed.kind == "future_delivery"


def test_job_event_rolls_back_when_required_outbox_conflicts(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    job = _job()
    store.create_job(
        job,
        outbox_events=(
            OutboxEvent(
                event_id="existing",
                job_id=job.job_id,
                kind="prepare",
                idempotency_key="duplicate-key",
                payload={"phase": "plan"},
            ),
        ),
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.apply_job_event_with_outbox(
            job_id=job.job_id,
            expected_version=job.version,
            event=JobEvent.CLAIM_PLAN,
            outbox_event=OutboxEvent(
                event_id="conflicting",
                job_id=job.job_id,
                kind="issue_comment",
                idempotency_key="duplicate-key",
                payload={"body": "claimed"},
            ),
        )

    unchanged = store.get_job(job.job_id)
    assert unchanged is not None
    assert (unchanged.status, unchanged.version) == (job.status, job.version)
    with store.connect() as connection:
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='JobClaimed'",
            (job.job_id,),
        ).fetchone()[0]
    assert audit_count == 0


def test_plan_reply_is_exactly_once_and_atomically_appended(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    sessions = SqliteSessionStore(store)
    session_id = str(uuid4())
    sessions.create(
        SessionMetadata(
            schema_version=6,
            session_id=session_id,
            workspace_root=str(tmp_path),
            model="model",
        ),
        agent_run_state(
            str(tmp_path),
            capabilities=CapabilityScope(allowed_tools=frozenset({"read_file"})),
        ),
    )
    job = _job().model_copy(update={"plan_session_id": session_id, "plan_workspace_ready": True})
    store.create_job(job)
    running = store.apply_job_event(
        job_id=job.job_id, expected_version=job.version, event=JobEvent.CLAIM_PLAN
    )
    blocked = store.apply_job_event_with_outbox(
        job_id=job.job_id, expected_version=running.version, event=JobEvent.PLAN_BLOCKED,
        outbox_event=OutboxEvent(event_id="blocked-comment", job_id=job.job_id, kind="issue_comment",
                                 idempotency_key="blocked-comment", payload={"body": "question"}),
    )
    with store.connect() as connection:
        payload = connection.execute(
            "SELECT payload_json FROM job_events WHERE job_id=? AND event_type='JobBlocked'",
            (job.job_id,),
        ).fetchone()
    assert payload is not None
    assert json.loads(payload[0])["job_event"] == JobEvent.PLAN_BLOCKED.value
    with pytest.raises(ValueError, match="CLEANUP_CONFLICT"):
        store.delete_terminal_job(blocked.job_id, expected_version=blocked.version)
    assert store.enqueue_plan_reply(
        source_id="github:123", job=blocked, body="answer",
        prepare_event=OutboxEvent(event_id="prepare-reply", job_id=job.job_id, kind="prepare",
                                  idempotency_key="prepare-reply", payload={"phase": "plan"}),
        acknowledgement_event=OutboxEvent(
            event_id="ack-reply", job_id=job.job_id, kind="issue_comment",
            idempotency_key="ack-reply", payload={"body": "received"},
        ),
    ) is True
    queued = store.get_job(job.job_id)
    assert queued is not None and queued.status == "queued_plan"
    assert queued.plan_workspace_ready is False
    assert store.append_external_message_once(
        source_id="github:123", session_id=session_id, text="answer"
    ) is True
    assert store.append_external_message_once(
        source_id="github:123", session_id=session_id, text="answer"
    ) is False
    snapshot = sessions.load(session_id)
    assert snapshot is not None and len(snapshot.messages) == 1
    assert snapshot.messages[0].content[0].text == "answer"


def test_sqlite_session_store_rejects_v5_metadata(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    sessions = SqliteSessionStore(store)
    sessions._append(
        "legacy",
        "metadata",
        json.dumps(
            {
                "schema_version": 5,
                "session_id": "legacy",
                "workspace_root": str(tmp_path),
                "model": "old",
                "system_prompt": "old",
            }
        ),
    )

    with pytest.raises(ValueError, match="Input should be 6"):
        sessions.load("legacy")


def test_plan_reply_does_not_recreate_a_missing_plan_session(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    session_id = str(uuid4())
    job = _job().model_copy(update={"plan_session_id": session_id})
    store.create_job(job)
    running = store.apply_job_event(
        job_id=job.job_id,
        expected_version=job.version,
        event=JobEvent.CLAIM_PLAN,
    )
    blocked = store.apply_job_event(
        job_id=job.job_id,
        expected_version=running.version,
        event=JobEvent.PLAN_BLOCKED,
    )
    store.enqueue_plan_reply(
        source_id="github:missing-session",
        job=blocked,
        body="answer",
        prepare_event=OutboxEvent(
            event_id="prepare-missing",
            job_id=job.job_id,
            kind="prepare",
            idempotency_key="prepare-missing",
            payload={"phase": "plan"},
        ),
        acknowledgement_event=OutboxEvent(
            event_id="ack-missing",
            job_id=job.job_id,
            kind="issue_comment",
            idempotency_key="ack-missing",
            payload={"body": "received"},
        ),
    )

    with pytest.raises(ValueError, match="Plan Session does not exist"):
        store.append_external_message_once(
            source_id="github:missing-session",
            session_id=session_id,
            text="answer",
        )
    assert store.pending_plan_reply(job.job_id) == ("github:missing-session", "answer")
