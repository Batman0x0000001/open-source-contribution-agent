"""使用 SQLite 持久化 Bot 作业、审批、Artifact 和 Outbox 数据。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator, Literal
from uuid import uuid4

from osc_agent.bot.domain.artifacts import DeliveryDraft, IssuePlanArtifact
from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.execution import BotApproval, ExecutionContract
from osc_agent.bot.domain.jobs import BotJob, JobStatus, utc_now
from osc_agent.bot.domain.state_machine import BotJobStateMachine
from osc_agent.bot.persistence.schema import (
    APPLICATION_VERSION,
    SCHEMA,
    SCHEMA_EPOCH,
    SCHEMA_MIGRATION_VERSION,
    STATE_MODEL_REVISION,
)
from osc_agent.runtime.messages import RuntimeMessage, TextBlock


class BotStore:
    """单机 SQLite 权威状态；所有写入都使用短事务和乐观版本。"""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def initialize(self) -> None:
        database_exists = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            legacy_jobs = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='bot_jobs'"
            ).fetchone()
            if legacy_jobs is not None:
                columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(bot_jobs)")}
                if "repository_id" not in columns:
                    raise ValueError("BOT_SCHEMA_EPOCH_MISMATCH: archive and reset Bot state")
            connection.executescript(SCHEMA)
            old = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            if old not in {None, SCHEMA_MIGRATION_VERSION}:
                raise ValueError("BOT_SCHEMA_EPOCH_MISMATCH: archive and reset Bot state")
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (SCHEMA_MIGRATION_VERSION, utc_now()),
            )
            connection.execute(
                """INSERT OR IGNORE INTO schema_meta(singleton, schema_epoch, state_model_revision,
                   application_version, created_at) VALUES(1, ?, ?, ?, ?)""",
                (SCHEMA_EPOCH, STATE_MODEL_REVISION, APPLICATION_VERSION, utc_now()),
            )
        if not database_exists:
            for candidate in (
                self.path,
                Path(str(self.path) + "-wal"),
                Path(str(self.path) + "-shm"),
            ):
                if candidate.exists():
                    candidate.chmod(0o660)
        self.check_schema()

    def check_schema(self) -> None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT schema_epoch, state_model_revision FROM schema_meta WHERE singleton=1"
            ).fetchone()
        if row != (SCHEMA_EPOCH, STATE_MODEL_REVISION):
            raise ValueError("BOT_SCHEMA_EPOCH_MISMATCH")

    def readiness_check(self) -> None:
        """Verify epoch compatibility and acquire a SQLite write lock without changing data."""

        self.check_schema()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("SELECT 1").fetchone()
            connection.rollback()

    def register_delivery(self, delivery_id: str, event_name: str) -> bool:
        if not delivery_id or not event_name:
            raise ValueError("delivery id and event name are required")
        try:
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO webhook_deliveries(delivery_id, event_name, received_at) VALUES(?, ?, ?)",
                    (delivery_id, event_name, utc_now()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def release_delivery(self, delivery_id: str) -> None:
        """仅在控制请求尚未产生任何状态变更就失败时允许 GitHub 重试。"""

        with self.connect() as connection:
            connection.execute(
                "DELETE FROM webhook_deliveries WHERE delivery_id=?", (delivery_id,)
            )

    def create_job(
        self,
        job: BotJob,
        *,
        input_value: dict[str, object] | None = None,
        outbox_event: OutboxEvent | None = None,
        outbox_events: tuple[OutboxEvent, ...] = (),
    ) -> None:
        encoded = job.model_dump_json()
        input_encoded: str | None = None
        if input_value is not None:
            input_encoded = json.dumps(input_value, ensure_ascii=False, sort_keys=True)
            if len(input_encoded) > 500_000:
                raise ValueError("GitHub issue evidence exceeds the bot input limit")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO bot_jobs(
                    job_id, repository_id, repository_full_name, issue_number, status, version,
                    lease_owner, lease_until, job_json, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.job_id,
                    job.repository_id,
                    job.repository_full_name,
                    job.issue_number,
                    job.status,
                    job.version,
                    job.lease_owner,
                    job.lease_until,
                    encoded,
                    job.created_at,
                    job.updated_at,
                ),
            )
            if input_encoded is not None:
                connection.execute(
                    "INSERT INTO bot_job_inputs(job_id, input_json, created_at) VALUES(?, ?, ?)",
                    (job.job_id, input_encoded, utc_now()),
                )
            initial_events = ((outbox_event,) if outbox_event is not None else ()) + outbox_events
            for initial_event in initial_events:
                if initial_event.job_id != job.job_id:
                    raise ValueError("initial outbox event belongs to a different job")
                connection.execute(
                    """INSERT INTO outbox_events(event_id, job_id, kind, idempotency_key,
                       status, attempts, next_attempt_at, event_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        initial_event.event_id,
                        initial_event.job_id,
                        initial_event.kind,
                        initial_event.idempotency_key,
                        initial_event.status,
                        initial_event.attempts,
                        initial_event.next_attempt_at,
                        initial_event.model_dump_json(),
                    ),
                )
            connection.execute(
                "INSERT INTO job_events(event_id, job_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    job.job_id,
                    "JobStarted",
                    json.dumps({"status": job.status}, sort_keys=True),
                    utc_now(),
                ),
            )
            connection.commit()

    def get_job(self, job_id: str) -> BotJob | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT job_json FROM bot_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return BotJob.model_validate_json(row[0]) if row else None

    def get_active_job(self, repository_id: int, issue_number: int) -> BotJob | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT job_json FROM bot_jobs WHERE issue_number=?
                   AND repository_id=?
                   AND status IN ('queued_plan','running_plan','blocked_plan','waiting_approval',
                     'queued_implementation','running_implementation','ready_to_publish','publishing','retry_wait')
                   ORDER BY created_at DESC LIMIT 1""",
                (issue_number, repository_id),
            ).fetchone()
        return BotJob.model_validate_json(row[0]) if row else None

    def save_execution_contract(self, contract: ExecutionContract) -> str:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO execution_contracts(
                   contract_hash, contract_version, contract_json, created_at) VALUES(?, ?, ?, ?)""",
                (contract.contract_hash, contract.contract_version, contract.model_dump_json(), utc_now()),
            )
        return contract.contract_hash

    def get_execution_contract(self, contract_hash: str) -> ExecutionContract | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT contract_json FROM execution_contracts WHERE contract_hash=?", (contract_hash,)
            ).fetchone()
        return ExecutionContract.model_validate_json(row[0]) if row else None

    def enqueue_plan_reply(
        self,
        *,
        source_id: str,
        job: BotJob,
        body: str,
        prepare_event: OutboxEvent,
        acknowledgement_event: OutboxEvent,
    ) -> bool:
        if job.status != "blocked_plan" or not job.plan_session_id:
            raise ValueError("BOT_INVALID_TRANSITION: reply requires a blocked Plan")
        if any(event.job_id != job.job_id for event in (prepare_event, acknowledgement_event)):
            raise ValueError("reply outbox event belongs to a different job")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            inserted = connection.execute(
                """INSERT OR IGNORE INTO bot_inbox_messages(
                   source_id, job_id, session_id, kind, body, status, created_at)
                   VALUES(?, ?, ?, 'plan_reply', ?, 'pending', ?)""",
                (source_id, job.job_id, job.plan_session_id, body, utc_now()),
            )
            if inserted.rowcount == 0:
                connection.rollback()
                return False
            updated = job.model_copy(update={
                "status": "queued_plan", "version": job.version + 1, "updated_at": utc_now(),
                "lease_owner": None, "lease_until": None,
                "plan_workspace_ready": False,
            })
            cursor = connection.execute(
                """UPDATE bot_jobs SET status=?, version=?, lease_owner=NULL, lease_until=NULL,
                   job_json=?, updated_at=? WHERE job_id=? AND version=?""",
                (updated.status, updated.version, updated.model_dump_json(), updated.updated_at,
                 job.job_id, job.version),
            )
            if cursor.rowcount != 1:
                raise ValueError("BOT_JOB_VERSION_CONFLICT")
            for event in (prepare_event, acknowledgement_event):
                connection.execute(
                    """INSERT INTO outbox_events(event_id, job_id, kind, idempotency_key,
                       status, attempts, next_attempt_at, event_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    (event.event_id, event.job_id, event.kind, event.idempotency_key,
                     event.status, event.attempts, event.next_attempt_at, event.model_dump_json()),
                )
            connection.execute(
                "INSERT INTO job_events(event_id, job_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (str(uuid4()), job.job_id, "PlanReplyReceived",
                 json.dumps({"source_id": source_id, "from": job.status, "to": updated.status}, sort_keys=True),
                 utc_now()),
            )
            connection.commit()
        return True

    def pending_plan_reply(self, job_id: str) -> tuple[str, str] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT source_id, body FROM bot_inbox_messages
                   WHERE job_id=? AND status='pending' ORDER BY created_at LIMIT 1""", (job_id,)
            ).fetchone()
        return (str(row[0]), str(row[1])) if row else None

    def consume_plan_reply(self, source_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE bot_inbox_messages SET status='consumed', consumed_at=?
                   WHERE source_id=? AND status='pending'""", (utc_now(), source_id)
            )

    def append_external_message_once(self, *, source_id: str, session_id: str, text: str) -> bool:
        """Append an inbox message and consume it in the same SQLite transaction."""

        message = RuntimeMessage(role="user", content=[TextBlock(text=text)])
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            inbox = connection.execute(
                """SELECT status, session_id FROM bot_inbox_messages
                   WHERE source_id=? AND kind='plan_reply'""", (source_id,),
            ).fetchone()
            if inbox is None:
                raise ValueError("unknown Plan reply")
            if str(inbox[1]) != session_id:
                raise ValueError("Plan reply belongs to a different Session")
            if inbox[0] == "consumed":
                connection.rollback()
                return False
            row = connection.execute(
                """SELECT sequence, event_id FROM session_records
                   WHERE session_id=? ORDER BY sequence DESC LIMIT 1""", (session_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Plan Session does not exist")
            connection.execute(
                """INSERT INTO session_records(session_id, sequence, event_id, previous_event_id,
                   record_type, payload_json, created_at) VALUES(?, ?, ?, ?, 'message', ?, ?)""",
                (session_id, int(row[0]) + 1, str(uuid4()), str(row[1]),
                 message.model_dump_json(), utc_now()),
            )
            changed = connection.execute(
                """UPDATE bot_inbox_messages SET status='consumed', consumed_at=?
                   WHERE source_id=? AND status='pending'""", (utc_now(), source_id),
            )
            if changed.rowcount != 1:
                raise ValueError("Plan reply consumption conflict")
            connection.commit()
        return True

    def update_job_fields(
        self,
        job_id: str,
        *,
        expected_version: int,
        **changes: object,
    ) -> BotJob:
        current = self.get_job(job_id)
        if current is None:
            raise ValueError(f"unknown bot job: {job_id}")
        if current.version != expected_version:
            raise ValueError("BOT_JOB_VERSION_CONFLICT")
        update = dict(changes)
        update["version"] = current.version + 1
        update["updated_at"] = utc_now()
        updated = BotJob.model_validate(
            {**current.model_dump(mode="json"), **update}
        )
        encoded = updated.model_dump_json()
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE bot_jobs SET status = ?, version = ?, lease_owner = ?,
                    lease_until = ?, job_json = ?, updated_at = ?
                    WHERE job_id = ? AND version = ?""",
                (
                    updated.status,
                    updated.version,
                    updated.lease_owner,
                    updated.lease_until,
                    encoded,
                    updated.updated_at,
                    job_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("BOT_JOB_VERSION_CONFLICT")
        return updated

    def transition(
        self,
        *,
        job_id: str,
        expected_version: int,
        status: JobStatus,
        **changes: object,
    ) -> BotJob:
        """Atomically validate and persist a state transition plus its audit event."""

        current = self.get_job(job_id)
        if current is None or current.version != expected_version:
            raise ValueError("BOT_JOB_VERSION_CONFLICT")
        _validate_transition(current.status, status, retry_phase=current.retry_phase)
        updated = BotJob.model_validate({
            **current.model_dump(mode="json"), **changes, "status": status,
            "version": current.version + 1, "updated_at": utc_now(),
        })
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE bot_jobs SET status=?, version=?, lease_owner=?, lease_until=?,
                   job_json=?, updated_at=? WHERE job_id=? AND version=?""",
                (updated.status, updated.version, updated.lease_owner, updated.lease_until,
                 updated.model_dump_json(), updated.updated_at, job_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise ValueError("BOT_JOB_VERSION_CONFLICT")
            connection.execute(
                "INSERT INTO job_events(event_id, job_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (str(uuid4()), job_id, _status_event(status),
                 json.dumps({"from": current.status, "to": status}, sort_keys=True), utc_now()),
            )
            connection.commit()
        return updated

    def claim_job(
        self,
        worker_id: str,
        *,
        phase: Literal["plan", "implementation"] | None = None,
        lease_seconds: int = 60,
    ) -> BotJob | None:
        now = datetime.now(timezone.utc)
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT candidate.job_json FROM bot_jobs AS candidate
                   WHERE candidate.status IN (
                     'queued_plan', 'running_plan',
                     'queued_implementation', 'running_implementation'
                   )
                     AND (
                       ? IS NULL
                       OR (? = 'plan' AND candidate.status IN ('queued_plan','running_plan'))
                       OR (? = 'implementation' AND candidate.status IN ('queued_implementation','running_implementation'))
                     )
                     AND (
                       (candidate.status IN ('queued_plan','running_plan')
                        AND json_extract(candidate.job_json, '$.plan_workspace_ready') = 1)
                       OR
                       (candidate.status IN ('queued_implementation','running_implementation')
                        AND json_extract(candidate.job_json, '$.implementation_workspace_ready') = 1)
                     )
                     AND (candidate.lease_until IS NULL OR candidate.lease_until < ?)
                     AND (
                       candidate.status NOT IN ('queued_implementation', 'running_implementation')
                       OR NOT EXISTS (
                         SELECT 1 FROM bot_jobs AS active
                         WHERE active.repository_full_name = candidate.repository_full_name
                           AND active.status = 'running_implementation'
                           AND active.job_id != candidate.job_id
                           AND active.lease_until >= ?
                       )
                     )
                   ORDER BY candidate.created_at LIMIT 1""",
                (phase, phase, phase, now.isoformat(), now.isoformat()),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            job = BotJob.model_validate_json(row[0])
            running_status: JobStatus = (
                "running_plan"
                if job.status in {"queued_plan", "running_plan"}
                else "running_implementation"
            )
            updated = job.model_copy(
                update={
                    "status": running_status,
                    ("plan_attempts" if running_status == "running_plan" else "implementation_attempts"):
                        (job.plan_attempts + 1 if running_status == "running_plan" else job.implementation_attempts + 1),
                    "lease_owner": worker_id,
                    "lease_until": lease_until,
                    "version": job.version + 1,
                    "updated_at": utc_now(),
                }
            )
            cursor = connection.execute(
                """UPDATE bot_jobs SET status=?, version=?, lease_owner=?, lease_until=?,
                   job_json=?, updated_at=? WHERE job_id=? AND version=?""",
                (
                    updated.status,
                    updated.version,
                    worker_id,
                    lease_until,
                    updated.model_dump_json(),
                    updated.updated_at,
                    job.job_id,
                    job.version,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.execute(
                "INSERT INTO job_events(event_id, job_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (str(uuid4()), job.job_id, "SessionStarted",
                 json.dumps({"from": job.status, "to": running_status}, sort_keys=True), utc_now()),
            )
            connection.commit()
        return updated

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: int = 60) -> BotJob:
        job = self.get_job(job_id)
        if job is None or job.lease_owner != worker_id:
            raise ValueError("BOT_JOB_LEASE_LOST")
        return self.update_job_fields(
            job_id,
            expected_version=job.version,
            lease_until=(datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat(),
        )

    def record_progress(self, job_id: str, event_type: str) -> BotJob:
        """Persist a bounded progress marker without RuntimeEvent payload content."""

        if not event_type or len(event_type) > 64:
            raise ValueError("invalid runtime progress event type")
        current = self.get_job(job_id)
        if current is None:
            raise ValueError("BOT_JOB_NOT_FOUND")
        return self.update_job_fields(
            job_id,
            expected_version=current.version,
            last_progress_event=event_type,
            last_progress_at=utc_now(),
        )

    def save_approval(self, approval: BotApproval) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO bot_approvals(approval_id, job_id, approval_json, created_at) VALUES(?, ?, ?, ?)",
                (approval.approval_id, approval.job_id, approval.model_dump_json(), approval.created_at),
            )
        self.append_job_event(approval.job_id, "ApprovalRecorded", {"approval_id": approval.approval_id})

    def get_approval(self, approval_id: str) -> BotApproval | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT approval_json FROM bot_approvals WHERE approval_id=?", (approval_id,)
            ).fetchone()
        return BotApproval.model_validate_json(row[0]) if row else None

    def approve_implementation(
        self,
        *,
        job_id: str,
        expected_version: int,
        approval: BotApproval,
        outbox_event: OutboxEvent,
    ) -> BotJob:
        current = self.get_job(job_id)
        if current is None or current.version != expected_version:
            raise ValueError("BOT_JOB_VERSION_CONFLICT")
        if approval.job_id != job_id or outbox_event.job_id != job_id:
            raise ValueError("approval transaction contains a mismatched job id")
        _validate_transition(current.status, "queued_implementation", retry_phase=current.retry_phase)
        updated = BotJob.model_validate(
            {
                **current.model_dump(mode="json"),
                "status": "queued_implementation",
                "approval_id": approval.approval_id,
                "implementation_workspace_path": None,
                "implementation_workspace_ready": False,
                "version": current.version + 1,
                "updated_at": utc_now(),
            }
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE bot_jobs SET status=?, version=?, lease_owner=?, lease_until=?,
                   job_json=?, updated_at=? WHERE job_id=? AND version=?""",
                (
                    updated.status,
                    updated.version,
                    updated.lease_owner,
                    updated.lease_until,
                    updated.model_dump_json(),
                    updated.updated_at,
                    job_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("BOT_JOB_VERSION_CONFLICT")
            connection.execute(
                "INSERT INTO bot_approvals(approval_id, job_id, approval_json, created_at) VALUES(?, ?, ?, ?)",
                (
                    approval.approval_id,
                    job_id,
                    approval.model_dump_json(),
                    approval.created_at,
                ),
            )
            connection.execute(
                """INSERT INTO outbox_events(event_id, job_id, kind, idempotency_key,
                   status, attempts, next_attempt_at, event_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    outbox_event.event_id,
                    job_id,
                    outbox_event.kind,
                    outbox_event.idempotency_key,
                    outbox_event.status,
                    outbox_event.attempts,
                    outbox_event.next_attempt_at,
                    outbox_event.model_dump_json(),
                ),
            )
            for event_type, payload in (
                ("ApprovalRecorded", {"approval_id": approval.approval_id}),
                ("JobStatusChanged", {"from": current.status, "to": updated.status}),
            ):
                connection.execute(
                    "INSERT INTO job_events(event_id, job_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                    (str(uuid4()), job_id, event_type, json.dumps(payload, sort_keys=True), utc_now()),
                )
            connection.commit()
        return updated

    def save_artifact(
        self,
        job_id: str,
        artifact: IssuePlanArtifact | DeliveryDraft,
    ) -> str:
        artifact_id = str(uuid4())
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO bot_artifacts(artifact_id, job_id, kind, artifact_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (artifact_id, job_id, artifact.evidence_type, artifact.model_dump_json(), utc_now()),
            )
        return artifact_id

    def save_job_input(self, job_id: str, value: dict[str, object]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if len(encoded) > 500_000:
            raise ValueError("GitHub issue evidence exceeds the bot input limit")
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO bot_job_inputs(job_id, input_json, created_at) VALUES(?, ?, ?)",
                (job_id, encoded, utc_now()),
            )

    def get_job_input(self, job_id: str) -> dict[str, object]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT input_json FROM bot_job_inputs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise ValueError("bot job has no GitHub issue evidence")
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ValueError("bot job input is invalid")
        return value

    def get_plan_artifact(self, artifact_id: str) -> IssuePlanArtifact | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT artifact_json FROM bot_artifacts WHERE artifact_id=? AND kind='issue_plan'",
                (artifact_id,),
            ).fetchone()
        return IssuePlanArtifact.model_validate_json(row[0]) if row else None

    def latest_plan_artifact(self, job_id: str) -> tuple[str, IssuePlanArtifact] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT artifact_id, artifact_json FROM bot_artifacts
                   WHERE job_id=? AND kind='issue_plan' ORDER BY created_at DESC LIMIT 1""",
                (job_id,),
            ).fetchone()
        return (str(row[0]), IssuePlanArtifact.model_validate_json(row[1])) if row else None

    def get_delivery_draft(self, job_id: str) -> DeliveryDraft | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT artifact_json FROM bot_artifacts
                   WHERE job_id=? AND kind='delivery_draft' ORDER BY created_at DESC LIMIT 1""",
                (job_id,),
            ).fetchone()
        return DeliveryDraft.model_validate_json(row[0]) if row else None

    def enqueue_outbox(self, event: OutboxEvent) -> bool:
        try:
            with self.connect() as connection:
                connection.execute(
                    """INSERT INTO outbox_events(event_id, job_id, kind, idempotency_key,
                       status, attempts, next_attempt_at, event_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.event_id,
                        event.job_id,
                        event.kind,
                        event.idempotency_key,
                        event.status,
                        event.attempts,
                        event.next_attempt_at,
                        event.model_dump_json(),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def transition_with_outbox(
        self,
        *,
        job_id: str,
        expected_version: int,
        status: JobStatus,
        event: OutboxEvent,
        **changes: object,
    ) -> BotJob:
        current = self.get_job(job_id)
        if current is None or current.version != expected_version:
            raise ValueError("BOT_JOB_VERSION_CONFLICT")
        if event.job_id != job_id:
            raise ValueError("outbox event belongs to a different job")
        _validate_transition(current.status, status, retry_phase=current.retry_phase)
        updated = BotJob.model_validate(
            {
                **current.model_dump(mode="json"),
                **changes,
                "status": status,
                "version": current.version + 1,
                "updated_at": utc_now(),
            }
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE bot_jobs SET status=?, version=?, lease_owner=?, lease_until=?,
                   job_json=?, updated_at=? WHERE job_id=? AND version=?""",
                (
                    updated.status,
                    updated.version,
                    updated.lease_owner,
                    updated.lease_until,
                    updated.model_dump_json(),
                    updated.updated_at,
                    job_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("BOT_JOB_VERSION_CONFLICT")
            connection.execute(
                """INSERT OR IGNORE INTO outbox_events(event_id, job_id, kind, idempotency_key,
                   status, attempts, next_attempt_at, event_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.job_id,
                    event.kind,
                    event.idempotency_key,
                    event.status,
                    event.attempts,
                    event.next_attempt_at,
                    event.model_dump_json(),
                ),
            )
            connection.execute(
                "INSERT INTO job_events(event_id, job_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    job_id,
                    _status_event(status),
                    json.dumps({"from": current.status, "to": status}, sort_keys=True),
                    utc_now(),
                ),
            )
            connection.commit()
        return updated

    def next_outbox(self, *, lease_seconds: int = 60) -> OutboxEvent | None:
        now = datetime.now(timezone.utc)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT event_id, event_json FROM outbox_events
                   WHERE status IN ('pending', 'failed', 'processing') AND next_attempt_at <= ?
                   ORDER BY next_attempt_at LIMIT 1""",
                (now.isoformat(),),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            event = OutboxEvent.model_validate_json(row[1]).model_copy(
                update={
                    "status": "processing",
                    "attempts": OutboxEvent.model_validate_json(row[1]).attempts + 1,
                    "next_attempt_at": (now + timedelta(seconds=lease_seconds)).isoformat(),
                    "last_error": None,
                }
            )
            cursor = connection.execute(
                """UPDATE outbox_events SET status='processing', attempts=?, next_attempt_at=?,
                   event_json=? WHERE event_id=? AND status IN ('pending', 'failed', 'processing')""",
                (event.attempts, event.next_attempt_at, event.model_dump_json(), row[0]),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        return event

    def complete_outbox(self, event_id: str) -> None:
        self._update_outbox(event_id, "completed", None)

    def fail_outbox(self, event_id: str, error: str, *, delay_seconds: int = 30) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT attempts FROM outbox_events WHERE event_id=?", (event_id,)).fetchone()
        attempts = int(row[0]) if row else 1
        if attempts >= 5:
            self.dead_letter_outbox(event_id, error)
            return
        backoff = min(delay_seconds * (2 ** max(0, attempts - 1)), 900)
        self._update_outbox(event_id, "failed", error[:1_000], delay_seconds=backoff)

    def dead_letter_outbox(self, event_id: str, error: str) -> None:
        self._update_outbox(event_id, "dead_letter", error[:1_000])

    def append_job_event(self, job_id: str, event_type: str, payload: dict[str, object]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO job_events(event_id, job_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (str(uuid4()), job_id, event_type, json.dumps(payload, sort_keys=True), utc_now()),
            )

    def terminal_jobs_before(self, cutoff: datetime) -> list[BotJob]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT job_json FROM bot_jobs
                   WHERE status IN ('completed', 'stale', 'dead_letter', 'cancelled')
                     AND updated_at < ? ORDER BY updated_at""",
                (cutoff.astimezone(timezone.utc).isoformat(),),
            ).fetchall()
        return [BotJob.model_validate_json(row[0]) for row in rows]

    def delete_terminal_job(self, job_id: str, *, expected_version: int) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        if job.version != expected_version or job.status not in {
            "completed", "blocked_plan", "stale", "dead_letter", "cancelled"
        }:
            raise ValueError("BOT_JOB_CLEANUP_CONFLICT")
        session_ids = [
            value
            for value in (job.plan_session_id, job.implementation_session_id)
            if value is not None
        ]
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT version, status FROM bot_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if current != (expected_version, job.status):
                connection.rollback()
                raise ValueError("BOT_JOB_CLEANUP_CONFLICT")
            for session_id in session_ids:
                connection.execute("DELETE FROM session_leases WHERE session_id=?", (session_id,))
                connection.execute("DELETE FROM session_records WHERE session_id=?", (session_id,))
            for table in (
                "outbox_events",
                "job_events",
                "bot_approvals",
                "bot_artifacts",
                "bot_job_inputs",
            ):
                connection.execute(f"DELETE FROM {table} WHERE job_id=?", (job_id,))
            connection.execute(
                "DELETE FROM bot_jobs WHERE job_id=? AND version=?", (job_id, expected_version)
            )
            connection.commit()

    def delete_deliveries_before(self, cutoff: datetime) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM webhook_deliveries WHERE received_at < ?",
                (cutoff.astimezone(timezone.utc).isoformat(),),
            )
        return cursor.rowcount

    def _update_outbox(
        self,
        event_id: str,
        status: Literal["completed", "failed", "dead_letter"],
        error: str | None,
        *,
        delay_seconds: int = 0,
    ) -> None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT event_json FROM outbox_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown outbox event: {event_id}")
            event = OutboxEvent.model_validate_json(row[0]).model_copy(
                update={
                    "status": status,
                    "last_error": error,
                    "next_attempt_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
                    ).isoformat(),
                }
            )
            connection.execute(
                """UPDATE outbox_events SET status=?, attempts=?, next_attempt_at=?, event_json=?
                   WHERE event_id=?""",
                (status, event.attempts, event.next_attempt_at, event.model_dump_json(), event_id),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()
def _status_event(status: JobStatus) -> str:
    return {
        "blocked_plan": "JobBlocked",
        "ready_to_publish": "DeliveryReady",
        "publishing": "PublishStarted",
        "completed": "JobCompleted",
        "dead_letter": "JobFailed",
        "cancelled": "JobCancelled",
    }.get(status, "JobStatusChanged")


def _validate_transition(current: str, target: str, *, retry_phase: str | None) -> None:
    if current == target:
        return
    event = {
        ("queued_plan", "running_plan"): "claim_plan",
        ("running_plan", "waiting_approval"): "plan_ready",
        ("running_plan", "blocked_plan"): "plan_blocked",
        ("blocked_plan", "queued_plan"): "reply_received",
        ("waiting_approval", "queued_implementation"): "implementation_approved",
        ("queued_implementation", "running_implementation"): "claim_implementation",
        ("running_implementation", "ready_to_publish"): "implementation_ready",
        ("ready_to_publish", "publishing"): "begin_publish",
        ("publishing", "completed"): "publish_complete",
    }.get((current, target))
    if target == "cancelled":
        event = "cancel"
    elif target == "stale":
        event = "mark_stale"
    elif target == "retry_wait":
        event = "schedule_retry"
    elif target == "dead_letter" and current == "retry_wait":
        event = "exhaust_retries"
    elif current == "retry_wait":
        event = "resume_retry"
    if event is None or BotJobStateMachine.transition(current, event, retry_phase=retry_phase) != target:
        raise ValueError(f"BOT_INVALID_TRANSITION:{current}:{target}")


