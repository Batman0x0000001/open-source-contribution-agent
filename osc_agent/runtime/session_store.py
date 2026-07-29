"""持久化会话记录和可延迟读取的工具结果。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Annotated, ContextManager, Iterator, Literal, Protocol, TypeAlias

import portalocker
from pydantic import Field

from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.session import (
    SessionMetadata,
    SessionOverview,
    SessionRuntimeState,
    SessionSnapshot,
)


class ToolResultStore(Protocol):
    def persist(
        self,
        *,
        session_id: str,
        tool_use_id: str,
        content: str,
    ) -> str: ...

    def read(self, *, session_id: str, result_id: str) -> str: ...


class MemoryToolResultStore:
    """测试和未配置持久层时的进程内保底实现。"""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}

    def persist(self, *, session_id: str, tool_use_id: str, content: str) -> str:
        result_id = tool_use_id
        self._values[(session_id, result_id)] = content
        return result_id

    def read(self, *, session_id: str, result_id: str) -> str:
        try:
            return self._values[(session_id, result_id)]
        except KeyError as exc:
            raise ValueError("unknown tool result for this session") from exc


class SessionStore(Protocol):
    def lease(self, session_id: str) -> ContextManager[None]: ...

    def create(self, metadata: SessionMetadata) -> None: ...

    def append_message(self, session_id: str, message: RuntimeMessage) -> None: ...

    def save_state(self, session_id: str, state: SessionRuntimeState) -> None: ...

    def load(self, session_id: str) -> SessionSnapshot | None: ...

    def list_overviews(self, *, limit: int | None = None) -> list[SessionOverview]: ...

    def latest_session_id(self) -> str | None: ...


class _MetadataRecord(ContractModel):
    type: Literal["metadata"] = "metadata"
    metadata: SessionMetadata


class _MessageRecord(ContractModel):
    type: Literal["message"] = "message"
    message: RuntimeMessage


class _StateRecord(ContractModel):
    type: Literal["state"] = "state"
    state: SessionRuntimeState


_SessionRecord: TypeAlias = Annotated[
    _MetadataRecord | _MessageRecord | _StateRecord,
    Field(discriminator="type"),
]


class _RecordEnvelope(ContractModel):
    record: _SessionRecord


class FileSessionStore:
    """以追加式 JSONL 保存权威会话；恢复时严格重放并验证每条记录。"""

    def __init__(self, sessions_root: Path) -> None:
        self.root = sessions_root.resolve()
        self._lock = threading.RLock()

    @contextmanager
    def lease(self, session_id: str) -> Iterator[None]:
        lock_root = self.root / ".locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{_safe_name(session_id)}.lock"
        try:
            with portalocker.Lock(
                lock_path,
                mode="a",
                timeout=0,
                flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            ):
                yield
        except portalocker.exceptions.LockException as exc:
            raise ValueError(f"SESSION_IN_USE: session is active in another process: {session_id}") from exc

    def create(self, metadata: SessionMetadata) -> None:
        path = self._path(metadata.session_id)
        with self._lock:
            if path.exists():
                raise ValueError(f"session already exists: {metadata.session_id}")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._append(path, _MetadataRecord(metadata=metadata))

    def append_message(self, session_id: str, message: RuntimeMessage) -> None:
        with self._lock:
            self._require_session(session_id)
            self._append(self._path(session_id), _MessageRecord(message=message))

    def save_state(self, session_id: str, state: SessionRuntimeState) -> None:
        with self._lock:
            self._require_session(session_id)
            self._append(self._path(session_id), _StateRecord(state=state))

    def load(self, session_id: str) -> SessionSnapshot | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        metadata: SessionMetadata | None = None
        messages: list[RuntimeMessage] = []
        state = SessionRuntimeState()
        with self._lock:
            self._recover_partial_tail(path)
        with self._lock, path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw_record = json.loads(line)
                    if (
                        isinstance(raw_record, dict)
                        and raw_record.get("type") == "metadata"
                        and (
                            not isinstance(raw_record.get("metadata"), dict)
                            or raw_record["metadata"].get("schema_version") != 4
                        )
                    ):
                        raise ValueError(
                            "unsupported session schema version; V4 requires schema_version=4"
                        )
                    record = _RecordEnvelope.model_validate_json(
                        json.dumps({"record": raw_record}, ensure_ascii=False)
                    ).record
                except Exception as exc:
                    raise ValueError(f"invalid session record at line {line_number}: {exc}") from exc
                if isinstance(record, _MetadataRecord):
                    if metadata is not None:
                        raise ValueError("session contains multiple metadata records")
                    metadata = record.metadata
                elif isinstance(record, _MessageRecord):
                    messages.append(record.message)
                else:
                    state = record.state
        if metadata is None:
            raise ValueError("session metadata is missing")
        return SessionSnapshot(metadata=metadata, messages=messages, runtime_state=state)

    def list_overviews(self, *, limit: int | None = None) -> list[SessionOverview]:
        if limit is not None and limit < 1:
            raise ValueError("session list limit must be at least 1")
        if not self.root.is_dir():
            return []
        paths = sorted(
            self.root.glob("*.jsonl"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        overviews: list[SessionOverview] = []
        for path in paths:
            updated_at = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
            try:
                snapshot = self.load(path.stem)
                if snapshot is None:
                    continue
                state = snapshot.runtime_state
                overviews.append(
                    SessionOverview(
                        session_id=path.stem,
                        model=snapshot.metadata.model,
                        repository_root=snapshot.metadata.repository_root,
                        updated_at=updated_at,
                        status=state.last_status or "unknown",
                        working_directory=(
                            state.worktree.path
                            if state.worktree is not None
                            else snapshot.metadata.initial_working_directory
                        ),
                        worktree=state.worktree,
                    )
                )
            except (OSError, ValueError) as exc:
                overviews.append(
                    SessionOverview(
                        session_id=path.stem,
                        updated_at=updated_at,
                        status="invalid",
                        error=str(exc),
                    )
                )
            if limit is not None and len(overviews) >= limit:
                break
        return overviews

    def latest_session_id(self) -> str | None:
        for overview in self.list_overviews():
            if overview.status != "invalid":
                return overview.session_id
        return None

    @staticmethod
    def _recover_partial_tail(path: Path) -> None:
        raw = path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        _prefix, separator, tail = raw.rpartition(b"\n")
        try:
            json.loads(tail.decode("utf-8"))
            with path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            return
        except (UnicodeDecodeError, json.JSONDecodeError):
            if not separator:
                raise ValueError("session contains an incomplete first record")
        recovery = path.with_suffix(path.suffix + ".corrupt-tail")
        suffix = 1
        while recovery.exists():
            recovery = path.with_suffix(path.suffix + f".corrupt-tail-{suffix}")
            suffix += 1
        recovery.write_bytes(tail)
        with path.open("wb") as stream:
            stream.write(raw[: len(raw) - len(tail)])
            stream.flush()
            os.fsync(stream.fileno())

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_safe_name(session_id)}.jsonl"

    def _require_session(self, session_id: str) -> None:
        if not self._path(session_id).is_file():
            raise ValueError(f"unknown session: {session_id}")

    @staticmethod
    def _append(path: Path, record: _SessionRecord) -> None:
        encoded = record.model_dump_json()
        # 再验证序列化结果，防止落盘格式与读取契约漂移。
        _RecordEnvelope.model_validate_json(
            json.dumps({"record": json.loads(encoded)}, ensure_ascii=False)
        )
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class FileToolResultStore:
    """将超出上下文预算的完整 Tool 输出保存在会话目录中。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def persist(
        self,
        *,
        session_id: str,
        tool_use_id: str,
        content: str,
    ) -> str:
        directory = self.root / _safe_name(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_name(tool_use_id)}.txt"
        suffix = 1
        while path.exists():
            path = directory / f"{_safe_name(tool_use_id)}-{suffix}.txt"
            suffix += 1
        path.write_text(content, encoding="utf-8")
        return path.stem

    def read(self, *, session_id: str, result_id: str) -> str:
        directory = (self.root / _safe_name(session_id)).resolve()
        path = (directory / f"{_safe_name(result_id)}.txt").resolve()
        if path.parent != directory or not path.is_file():
            raise ValueError("unknown tool result for this session")
        return path.read_text(encoding="utf-8")


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    if not safe or safe != value:
        raise ValueError("identifier may contain only letters, numbers, underscores, or hyphens")
    return safe
