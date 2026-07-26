from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from osc_agent.bot.models import BotJob, OutboxEvent
from osc_agent.bot.store import BotStore, SqliteSessionStore
from osc_agent.runtime.models import (
    CapabilityScope,
    RuntimeMessage,
    SessionMetadata,
    SessionRuntimeState,
    TextBlock,
)


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


def test_sqlite_job_delivery_version_and_claim(tmp_path: Path) -> None:
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    assert store.register_delivery("delivery-1", "issue_comment") is True
    assert store.register_delivery("delivery-1", "issue_comment") is False
    job = _job()
    store.create_job(job)
    with pytest.raises(ValueError, match="VERSION_CONFLICT"):
        store.update_job(job.job_id, expected_version=99, status="failed")
    assert store.claim_job("worker") is None  # Workspace 尚未由可信 Control 准备。
    prepared = store.update_job(
        job.job_id,
        expected_version=job.version,
        workspace_path=str(tmp_path / "workspace"),
    )
    claimed = store.claim_job("worker")
    assert claimed is not None
    assert claimed.status == "running_plan"
    assert claimed.lease_owner == "worker"
    assert claimed.version == prepared.version + 1


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
