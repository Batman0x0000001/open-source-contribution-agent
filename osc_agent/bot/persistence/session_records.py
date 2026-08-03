"""集中实现 Bot SQLite Session Record 的追加规则。"""

from __future__ import annotations

import sqlite3
from typing import Literal
from uuid import uuid4

from osc_agent.bot.domain.jobs import utc_now


SessionRecordType = Literal["metadata", "message", "state"]


def append_session_record(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    record_type: SessionRecordType,
    payload_json: str,
    require_existing: bool,
) -> None:
    """在调用方事务中追加一条严格链接的 Session Record。"""

    row = connection.execute(
        """SELECT sequence, event_id FROM session_records
           WHERE session_id=? ORDER BY sequence DESC LIMIT 1""",
        (session_id,),
    ).fetchone()
    if row is None and require_existing:
        raise ValueError("session does not exist")
    sequence = 1 if row is None else int(row[0]) + 1
    previous = None if row is None else str(row[1])
    connection.execute(
        """INSERT INTO session_records(session_id, sequence, event_id, previous_event_id,
           record_type, payload_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)""",
        (session_id, sequence, str(uuid4()), previous, record_type, payload_json, utc_now()),
    )
