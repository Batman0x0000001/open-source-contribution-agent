from __future__ import annotations

from pathlib import Path

import pytest

from osc_agent.runtime.models import CapabilityScope, RuntimeMessage, SessionMetadata, SessionRuntimeState, TextBlock
from osc_agent.runtime.session_store import FileSessionStore


def test_jsonl_session_create_append_state_and_resume(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    metadata = SessionMetadata(
        session_id="session-1",
        repository_root=str(tmp_path),
        initial_working_directory=str(tmp_path),
        model="test",
        system_prompt="saved prompt",
        capabilities=CapabilityScope(allowed_tools=frozenset({"read_file"})),
    )
    message = RuntimeMessage(role="user", content=[TextBlock(text="hello")])

    store.create(metadata)
    store.append_message("session-1", message)
    store.save_state("session-1", SessionRuntimeState(permission_mode="plan", plan_path="session-1.md"))
    snapshot = store.load("session-1")

    assert snapshot is not None
    assert snapshot.metadata == metadata
    assert snapshot.messages == [message]
    assert snapshot.runtime_state.permission_mode == "plan"


def test_jsonl_session_rejects_corrupt_records(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    store.create(SessionMetadata(session_id="session-1", repository_root=str(tmp_path), initial_working_directory=str(tmp_path), model="test"))
    path = tmp_path / "session-1.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"type":"message","message":{"role":"user","content":[]}}\n')

    with pytest.raises(ValueError, match="invalid session record"):
        store.load("session-1")


def test_session_identifier_cannot_escape_store(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    with pytest.raises(ValueError, match="identifier"):
        store.load("../escape")
