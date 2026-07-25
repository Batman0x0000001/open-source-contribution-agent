from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import Field

from osc_agent.runtime.models import (
    ContractModel,
    RuntimeMessage,
    SessionMetadata,
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


class SessionStore(Protocol):
    def create(self, metadata: SessionMetadata) -> None: ...

    def append_message(self, session_id: str, message: RuntimeMessage) -> None: ...

    def save_state(self, session_id: str, state: SessionRuntimeState) -> None: ...

    def load(self, session_id: str) -> SessionSnapshot | None: ...


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
        with self._lock, path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = _RecordEnvelope.model_validate_json(
                        json.dumps({"record": json.loads(line)}, ensure_ascii=False)
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
