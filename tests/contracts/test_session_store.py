"""验证 File SessionStore 的 V5 追加、恢复、拒绝旧版本和 Lease。"""

from __future__ import annotations

from pathlib import Path

import pytest

from osc_agent.runtime.messages import RuntimeMessage, TextBlock
from osc_agent.runtime.session import SessionMetadata
from osc_agent.runtime.session_store import FileSessionStore
from osc_agent.runtime.state import PlanModeEntered, PlanSaved
from tests.runtime_factories import agent_run_state


def metadata(session_id: str, root: Path) -> SessionMetadata:
    return SessionMetadata(
        schema_version=5,
        session_id=session_id,
        workspace_root=str(root),
        model="test",
        system_prompt="saved prompt",
    )


def test_jsonl_session_create_append_state_and_resume(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    state = agent_run_state(str(tmp_path)).apply(PlanModeEntered()).apply(
        PlanSaved(path="session-1.md")
    )
    store.create(metadata("session-1", tmp_path), state)
    message = RuntimeMessage(role="user", content=[TextBlock(text="hello")])
    store.append_message("session-1", message)
    store.save_state("session-1", state.with_status("completed", "end_turn"))

    snapshot = store.load("session-1")

    assert snapshot is not None
    assert snapshot.messages == [message]
    assert snapshot.state.permissions.mode == "plan"
    assert snapshot.state.permissions.plan_path == "session-1.md"
    assert snapshot.state.last_status == "completed"


def test_jsonl_session_rejects_corrupt_records(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    store.create(metadata("session-1", tmp_path), agent_run_state(str(tmp_path)))
    with (tmp_path / "session-1.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"type":"message","message":{"role":"user","content":[]}}\n')
    with pytest.raises(ValueError, match="invalid session record"):
        store.load("session-1")


def test_session_identifier_cannot_escape_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identifier"):
        FileSessionStore(tmp_path).load("../escape")


def test_v4_session_is_explicitly_rejected(tmp_path: Path) -> None:
    (tmp_path / "legacy.jsonl").write_text(
        '{"type":"metadata","metadata":{"schema_version":4,"session_id":"legacy",'
        '"repository_root":"C:/repo","model":"old"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="V5 requires schema_version=5"):
        FileSessionStore(tmp_path).load("legacy")


def test_session_lease_is_exclusive_across_store_instances(tmp_path: Path) -> None:
    first = FileSessionStore(tmp_path)
    second = FileSessionStore(tmp_path)
    with first.lease("session-1"):
        with pytest.raises(ValueError, match="SESSION_IN_USE"):
            with second.lease("session-1"):
                raise AssertionError("second lease must not be acquired")


def test_incomplete_final_record_is_quarantined(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    store.create(metadata("session-1", tmp_path), agent_run_state(str(tmp_path)))
    path = tmp_path / "session-1.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"type":"message","message":')
    assert store.load("session-1") is not None
    assert path.read_bytes().endswith(b"\n")
    assert (tmp_path / "session-1.jsonl.corrupt-tail").exists()


def test_valid_final_record_without_newline_is_normalized(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    store.create(metadata("session-1", tmp_path), agent_run_state(str(tmp_path)))
    path = tmp_path / "session-1.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert store.load("session-1") is not None
    store.append_message(
        "session-1", RuntimeMessage(role="user", content=[TextBlock(text="after")])
    )
    assert store.load("session-1").messages[0].content[0].text == "after"


def test_session_overviews_sort_and_skip_invalid_for_latest(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    for session_id in ("older", "newer"):
        store.create(
            metadata(session_id, tmp_path),
            agent_run_state(str(tmp_path), status="completed"),
        )
    (tmp_path / "invalid.jsonl").write_text("{broken\n", encoding="utf-8")
    overviews = store.list_overviews()
    assert {item.session_id for item in overviews} == {"older", "newer", "invalid"}
    assert next(item for item in overviews if item.session_id == "invalid").status == "invalid"
    assert store.latest_session_id() in {"older", "newer"}
