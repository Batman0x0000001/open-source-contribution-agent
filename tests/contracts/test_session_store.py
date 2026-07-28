"""验证会话持久化的契约、边界条件与回归行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from osc_agent.runtime.models import (
    CapabilityScope,
    CompletionRequirements,
    FileObservation,
    RepositoryInstructionState,
    RuntimeMessage,
    SessionMetadata,
    SessionRuntimeState,
    TextBlock,
)
from osc_agent.runtime.session_store import FileSessionStore


def test_jsonl_session_create_append_state_and_resume(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    metadata = SessionMetadata(
        schema_version=4,
        session_id="session-1",
        repository_root=str(tmp_path),
        initial_working_directory=str(tmp_path),
        model="test",
        system_prompt="saved prompt",
        capabilities=CapabilityScope(allowed_tools=frozenset({"read_file"})),
        completion_requirements=CompletionRequirements(
            required_evidence=frozenset({"successful_test"}),
            waivable_evidence=frozenset({"successful_test"}),
        ),
    )
    message = RuntimeMessage(role="user", content=[TextBlock(text="hello")])

    store.create(metadata)
    store.append_message("session-1", message)
    store.save_state(
        "session-1",
        SessionRuntimeState(
            permission_mode="plan",
            plan_path="session-1.md",
            instruction_state=RepositoryInstructionState(
                active_paths=["AGENTS.md"]
            ),
            file_observations={
                "x.py": FileObservation(
                    path="x.py",
                    content_hash="abc",
                    mtime_ns=1,
                    complete=True,
                )
            },
            completion_requirements=CompletionRequirements(
                required_evidence=frozenset(
                    {"successful_test", "git_change_snapshot"}
                ),
                waivable_evidence=frozenset({"successful_test"}),
            ),
        ),
    )
    snapshot = store.load("session-1")

    assert snapshot is not None
    assert snapshot.metadata == metadata
    assert snapshot.messages == [message]
    assert snapshot.runtime_state.permission_mode == "plan"
    assert snapshot.metadata.completion_requirements.required_evidence == {
        "successful_test"
    }
    assert snapshot.runtime_state.instruction_state.active_paths == ["AGENTS.md"]
    assert snapshot.runtime_state.file_observations["x.py"].complete is True
    assert snapshot.runtime_state.completion_requirements is not None
    assert snapshot.runtime_state.completion_requirements.required_evidence == {
        "successful_test",
        "git_change_snapshot",
    }


def test_jsonl_session_rejects_corrupt_records(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    store.create(SessionMetadata(schema_version=4, session_id="session-1", repository_root=str(tmp_path), initial_working_directory=str(tmp_path), model="test"))
    path = tmp_path / "session-1.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"type":"message","message":{"role":"user","content":[]}}\n')

    with pytest.raises(ValueError, match="invalid session record"):
        store.load("session-1")


def test_session_identifier_cannot_escape_store(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    with pytest.raises(ValueError, match="identifier"):
        store.load("../escape")


def test_legacy_session_without_v4_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        '{"type":"metadata","metadata":{"session_id":"legacy","repository_root":"C:/repo",'
        '"initial_working_directory":"C:/repo","model":"old"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="V4 requires schema_version=4"):
        FileSessionStore(tmp_path).load("legacy")


def test_session_lease_is_exclusive_across_store_instances(tmp_path: Path) -> None:
    first = FileSessionStore(tmp_path)
    second = FileSessionStore(tmp_path)

    with first.lease("session-1"):
        with pytest.raises(ValueError, match="SESSION_IN_USE"):
            with second.lease("session-1"):
                raise AssertionError("second lease must not be acquired")


def test_incomplete_final_jsonl_record_is_quarantined_and_removed(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path)
    store.create(
        SessionMetadata(
            schema_version=4,
            session_id="session-1",
            repository_root=str(tmp_path),
            initial_working_directory=str(tmp_path),
            model="test",
        )
    )
    path = tmp_path / "session-1.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"type":"message","message":')

    snapshot = store.load("session-1")

    assert snapshot is not None
    assert snapshot.messages == []
    assert path.read_bytes().endswith(b"\n")
    assert (tmp_path / "session-1.jsonl.corrupt-tail").read_bytes().startswith(
        b'{"type":"message"'
    )


def test_valid_final_record_without_newline_is_normalized_before_append(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path)
    store.create(
        SessionMetadata(
            schema_version=4,
            session_id="session-1",
            repository_root=str(tmp_path),
            initial_working_directory=str(tmp_path),
            model="test",
        )
    )
    path = tmp_path / "session-1.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    assert store.load("session-1") is not None
    store.append_message(
        "session-1",
        RuntimeMessage(role="user", content=[TextBlock(text="after recovery")]),
    )

    assert store.load("session-1").messages[0].content[0].text == "after recovery"


def test_session_overviews_are_sorted_and_latest_skips_invalid(
    tmp_path: Path,
) -> None:
    store = FileSessionStore(tmp_path)
    for session_id in ("older", "newer"):
        store.create(
            SessionMetadata(
                schema_version=4,
                session_id=session_id,
                repository_root=str(tmp_path),
                initial_working_directory=str(tmp_path),
                model="test",
            )
        )
        store.save_state(
            session_id,
            SessionRuntimeState(last_status="completed", last_reason="end_turn"),
        )
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    older.touch()
    newer.touch()
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{broken\n", encoding="utf-8")
    invalid.touch()

    overviews = store.list_overviews()

    assert {item.session_id for item in overviews} == {"older", "newer", "invalid"}
    assert next(item for item in overviews if item.session_id == "invalid").status == "invalid"
    assert store.latest_session_id() in {"older", "newer"}


def test_v4_session_defaults_new_v8_runtime_fields(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path)
    store.create(
        SessionMetadata(
            schema_version=4,
            session_id="v4",
            repository_root=str(tmp_path),
            initial_working_directory=str(tmp_path),
            model="test",
        )
    )

    snapshot = store.load("v4")

    assert snapshot is not None
    assert snapshot.runtime_state.permission_grants == []
    assert snapshot.runtime_state.last_status is None
