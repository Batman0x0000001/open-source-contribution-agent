"""验证Bot SQLite Store的契约、边界条件与回归行为。"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest

from osc_agent.bot.models import BotJob, OutboxEvent
from osc_agent.bot.store import BotStore, SqliteSessionStore
from osc_agent.runtime.messages import RuntimeMessage, TextBlock
from osc_agent.runtime.session import SessionMetadata, SessionRuntimeState
from osc_agent.runtime.tool_models import CapabilityScope


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


def test_sqlite_job_delivery_version_and_claim(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    assert store.register_delivery("delivery-1", "issue_comment") is True
    assert store.register_delivery("delivery-1", "issue_comment") is False
    job = _job()
    store.create_job(job)
    with pytest.raises(ValueError, match="VERSION_CONFLICT"):
        store.update_job_fields(job.job_id, expected_version=99, error_code="failed")
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

    progressed = store.record_progress(job.job_id, "plan:model_request_started")
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
        schema_version=4,
        session_id=session_id,
        repository_root=str(tmp_path),
        initial_working_directory=str(tmp_path),
        model="model",
        capabilities=CapabilityScope(allowed_tools=frozenset({"read_file"})),
    )
    sessions.create(metadata)
    sessions.append_message(
        session_id,
        RuntimeMessage(role="user", content=[TextBlock(text="goal")]),
    )
    sessions.save_state(session_id, SessionRuntimeState(last_status="completed"))
    snapshot = sessions.load(session_id)
    assert snapshot is not None
    assert snapshot.messages[0].content[0].text == "goal"
    assert snapshot.runtime_state.last_status == "completed"
    with sessions.lease(session_id):
        with pytest.raises(ValueError, match="SESSION_IN_USE"):
            with sessions.lease(session_id):
                pass


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


def test_plan_reply_is_exactly_once_and_atomically_appended(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    sessions = SqliteSessionStore(store)
    session_id = str(uuid4())
    sessions.create(SessionMetadata(
        schema_version=4, session_id=session_id, repository_root=str(tmp_path),
        initial_working_directory=str(tmp_path), model="model",
        capabilities=CapabilityScope(allowed_tools=frozenset({"read_file"})),
    ))
    job = _job().model_copy(update={"plan_session_id": session_id, "plan_workspace_ready": True})
    store.create_job(job)
    running = store.transition(job_id=job.job_id, expected_version=job.version, status="running_plan")
    blocked = store.transition_with_outbox(
        job_id=job.job_id, expected_version=running.version, status="blocked_plan",
        event=OutboxEvent(event_id="blocked-comment", job_id=job.job_id, kind="issue_comment",
                          idempotency_key="blocked-comment", payload={"body": "question"}),
    )
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
