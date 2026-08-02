"""使用 Bot SQLite 数据库持久化 Runtime Session V6 记录。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator
from uuid import uuid4

from osc_agent.bot.persistence.session_records import SessionRecordType, append_session_record
from osc_agent.bot.persistence.store import BotStore
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.session import (
    SessionMetadata,
    SessionOverview,
    SessionSnapshot,
)
from osc_agent.runtime.session_store import SessionStore
from osc_agent.runtime.state import AgentRunState


class SqliteSessionStore(SessionStore):
    """为远程 Worker 提供追加式、可跨进程恢复的 Session Store。"""

    def __init__(self, store: BotStore, *, lease_seconds: int = 60) -> None:
        self.store = store
        self.lease_seconds = lease_seconds
        self._lease_owners: dict[str, str] = {}

    @contextmanager
    def lease(self, session_id: str) -> Iterator[None]:
        owner = str(uuid4())
        now = datetime.now(timezone.utc)
        until = (now + timedelta(seconds=self.lease_seconds)).isoformat()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner, lease_until FROM session_leases WHERE session_id=?", (session_id,)
            ).fetchone()
            if row is not None and row[1] >= now.isoformat():
                connection.rollback()
                raise ValueError(f"SESSION_IN_USE: session is active in another process: {session_id}")
            connection.execute(
                """INSERT INTO session_leases(session_id, owner, lease_until) VALUES(?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET owner=excluded.owner, lease_until=excluded.lease_until""",
                (session_id, owner, until),
            )
            connection.commit()
        self._lease_owners[session_id] = owner
        try:
            yield
        finally:
            if self._lease_owners.get(session_id) == owner:
                self._lease_owners.pop(session_id, None)
            with self.store.connect() as connection:
                connection.execute(
                    "DELETE FROM session_leases WHERE session_id=? AND owner=?", (session_id, owner)
                )

    def renew_lease(self, session_id: str) -> None:
        owner = self._lease_owners.get(session_id)
        if owner is None:
            raise ValueError("SESSION_LEASE_LOST")
        until = (
            datetime.now(timezone.utc) + timedelta(seconds=self.lease_seconds)
        ).isoformat()
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE session_leases SET lease_until=? WHERE session_id=? AND owner=?",
                (until, session_id, owner),
            )
            if cursor.rowcount != 1:
                raise ValueError("SESSION_LEASE_LOST")
            connection.commit()

    def create(self, metadata: SessionMetadata, state: AgentRunState) -> None:
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM session_records WHERE session_id=? LIMIT 1",
                (metadata.session_id,),
            ).fetchone()
            if exists is not None:
                raise ValueError(f"session already exists: {metadata.session_id}")
            append_session_record(
                connection,
                session_id=metadata.session_id,
                record_type="metadata",
                payload_json=metadata.model_dump_json(),
                require_existing=False,
            )
            append_session_record(
                connection,
                session_id=metadata.session_id,
                record_type="state",
                payload_json=state.model_dump_json(),
                require_existing=True,
            )
            connection.commit()

    def append_message(self, session_id: str, message: RuntimeMessage) -> None:
        self._require(session_id)
        self._append(session_id, "message", message.model_dump_json())

    def save_state(self, session_id: str, state: AgentRunState) -> None:
        self._require(session_id)
        self._append(session_id, "state", state.model_dump_json())

    def load(self, session_id: str) -> SessionSnapshot | None:
        with self.store.connect() as connection:
            rows = connection.execute(
                """SELECT sequence, event_id, previous_event_id, record_type, payload_json
                   FROM session_records WHERE session_id=? ORDER BY sequence""",
                (session_id,),
            ).fetchall()
        if not rows:
            return None
        metadata: SessionMetadata | None = None
        messages: list[RuntimeMessage] = []
        state: AgentRunState | None = None
        previous: str | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            sequence, event_id, previous_event_id, record_type, payload = row
            if sequence != expected_sequence or previous_event_id != previous:
                raise ValueError("invalid session event chain")
            previous = event_id
            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError("session contains multiple metadata records")
                metadata = SessionMetadata.model_validate_json(payload)
            elif record_type == "message":
                messages.append(RuntimeMessage.model_validate_json(payload))
            elif record_type == "state":
                state = AgentRunState.model_validate_json(payload)
            else:
                raise ValueError(f"unknown session record type: {record_type}")
        if metadata is None:
            raise ValueError("session metadata is missing")
        if state is None:
            raise ValueError("session state is missing")
        return SessionSnapshot(metadata=metadata, messages=messages, state=state)

    def list_overviews(self, *, limit: int | None = None) -> list[SessionOverview]:
        with self.store.connect() as connection:
            rows = connection.execute(
                """SELECT session_id, MAX(created_at) AS updated_at
                   FROM session_records GROUP BY session_id ORDER BY updated_at DESC"""
                + (" LIMIT ?" if limit is not None else ""),
                (() if limit is None else (limit,)),
            ).fetchall()
        overviews: list[SessionOverview] = []
        for session_id, updated_at in rows:
            try:
                snapshot = self.load(session_id)
                assert snapshot is not None
                state = snapshot.state
                overviews.append(
                    SessionOverview(
                        session_id=session_id,
                        model=snapshot.metadata.model,
                        workspace_root=snapshot.metadata.workspace_root,
                        updated_at=updated_at,
                        status=state.last_status or "unknown",
                        working_directory=state.workspace.working_directory,
                        worktree=state.workspace.worktree,
                    )
                )
            except ValueError as exc:
                overviews.append(
                    SessionOverview(
                        session_id=session_id,
                        updated_at=updated_at,
                        status="invalid",
                        error=str(exc),
                    )
                )
        return overviews

    def latest_session_id(self) -> str | None:
        return next(
            (item.session_id for item in self.list_overviews() if item.status != "invalid"),
            None,
        )

    def _require(self, session_id: str) -> None:
        if self.load(session_id) is None:
            raise ValueError(f"unknown session: {session_id}")

    def _append(
        self,
        session_id: str,
        record_type: SessionRecordType,
        payload_json: str,
    ) -> None:
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            append_session_record(
                connection,
                session_id=session_id,
                record_type=record_type,
                payload_json=payload_json,
                require_existing=record_type != "metadata",
            )
            connection.commit()
