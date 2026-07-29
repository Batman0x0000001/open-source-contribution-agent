"""使用 Bot SQLite 数据库持久化 Runtime Session V5 记录。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator
from uuid import uuid4

from osc_agent.bot.domain.jobs import utc_now
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

    def __init__(self, store: BotStore, *, lease_seconds: int = 3_600) -> None:
        self.store = store
        self.lease_seconds = lease_seconds

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
        try:
            yield
        finally:
            with self.store.connect() as connection:
                connection.execute(
                    "DELETE FROM session_leases WHERE session_id=? AND owner=?", (session_id, owner)
                )

    def create(self, metadata: SessionMetadata, state: AgentRunState) -> None:
        if self.load(metadata.session_id) is not None:
            raise ValueError(f"session already exists: {metadata.session_id}")
        self._append(metadata.session_id, "metadata", metadata.model_dump_json())
        self._append(metadata.session_id, "state", state.model_dump_json())

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
                """SELECT DISTINCT session_id FROM session_records
                   ORDER BY (SELECT MAX(created_at) FROM session_records s2
                             WHERE s2.session_id=session_records.session_id) DESC"""
                + (" LIMIT ?" if limit is not None else ""),
                (() if limit is None else (limit,)),
            ).fetchall()
        overviews: list[SessionOverview] = []
        for (session_id,) in rows:
            try:
                snapshot = self.load(session_id)
                assert snapshot is not None
                state = snapshot.state
                overviews.append(
                    SessionOverview(
                        session_id=session_id,
                        model=snapshot.metadata.model,
                        workspace_root=snapshot.metadata.workspace_root,
                        updated_at=utc_now(),
                        status=state.last_status or "unknown",
                        working_directory=state.workspace.working_directory,
                        worktree=state.workspace.worktree,
                    )
                )
            except ValueError as exc:
                overviews.append(
                    SessionOverview(
                        session_id=session_id,
                        updated_at=utc_now(),
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

    def _append(self, session_id: str, record_type: str, payload_json: str) -> None:
        with self.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT sequence, event_id FROM session_records
                   WHERE session_id=? ORDER BY sequence DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            sequence = 1 if row is None else int(row[0]) + 1
            previous = None if row is None else str(row[1])
            connection.execute(
                """INSERT INTO session_records(session_id, sequence, event_id, previous_event_id,
                   record_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (session_id, sequence, str(uuid4()), previous, record_type, payload_json, utc_now()),
            )
            connection.commit()
