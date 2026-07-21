from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from osc_agent.harness.command import CommandKind, classify_command
from osc_agent.harness.contracts import ToolResult, action_fingerprint
from osc_agent.harness.trace import sanitize_tool_arguments


RUNTIME_STATE_SCHEMA_VERSION = 3
MAX_RUNTIME_ITEMS = 20
_runtime_state_lock = threading.RLock()


class FailedStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    fingerprint: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    summary: str = ""
    occurrences: int = Field(default=1, ge=1)


class TestObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    command: str
    ok: bool
    error_code: str | None = None
    artifact_path: str | None = None


class RuntimeState(BaseModel):
    """跨轮次保存的最小运行摘要；Contribution Run 仍是工作流权威数据源。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)

    schema_version: Literal[3] = RUNTIME_STATE_SCHEMA_VERSION
    active_run_id: str | None = None
    objective: str = ""
    current_instruction: str = ""
    scope_exclusions: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    forbidden_paths: list[str] = Field(default_factory=list)
    integrity_anchors: dict[str, str] = Field(default_factory=dict)
    acceptance_checks: list[dict[str, Any]] = Field(default_factory=list)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    recent_test_observations: list[TestObservation] = Field(default_factory=list)
    failed_strategies: list[FailedStrategy] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

    @field_validator("unresolved_questions")
    @classmethod
    def normalize_unresolved_questions(cls, values: list[str]) -> list[str]:
        deduplicated: list[str] = []
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in deduplicated:
                deduplicated.append(normalized)
        return deduplicated[-MAX_RUNTIME_ITEMS:]


def runtime_state_path(repo_root: Path, session_id: str = "default") -> Path:
    return repo_root / ".osc_agent" / "sessions" / session_id / "runtime_state.json"


def load_runtime_state(repo_root: Path, *, session_id: str = "default") -> RuntimeState:
    with _runtime_state_lock:
        path = runtime_state_path(repo_root, session_id)
        if not path.exists():
            return RuntimeState()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _reset_state("runtime state could not be parsed and was reset")
        if not isinstance(data, dict):
            return _reset_state("runtime state root must be an object and was reset")
        if data.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION:
            return _reset_state("runtime state schema is incompatible and was reset")
        try:
            return RuntimeState.model_validate(data)
        except ValidationError:
            return _reset_state("runtime state fields are incompatible and were reset")


def save_runtime_state(repo_root: Path, state: RuntimeState, *, session_id: str = "default") -> None:
    with _runtime_state_lock:
        path = runtime_state_path(repo_root, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            temp.write_text(
                json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                # 原子替换成功后临时文件通常已不存在；清理失败不应覆盖原始写入结果。
                pass


def refresh_runtime_state(
    repo_root: Path,
    current_instruction: str,
    *,
    objective: str | None = None,
    run_id: str | None = None,
    session_id: str = "default",
) -> RuntimeState:
    with _runtime_state_lock:
        # 先验证 Run，再修改已有状态，避免无效标识清空或污染会话。
        run_dir = _run_dir(repo_root, run_id)
        state = load_runtime_state(repo_root, session_id=session_id)
        if objective:
            state.objective = objective
        elif current_instruction and not state.objective:
            state.objective = current_instruction
        if current_instruction:
            state.current_instruction = current_instruction

        if state.active_run_id != run_id:
            _clear_run_state(state)
            state.active_run_id = run_id

        if run_dir is not None:
            run = _read_json(run_dir / "run.json")
            design = _read_json(run_dir / "02_design.json")
            implementation = _read_json(run_dir / "03_implementation.json")
            state.scope_exclusions = [str(item) for item in design.get("out_of_scope") or []]
            state.allowed_files = [str(item) for item in design.get("allowed_files") or []]
            state.forbidden_paths = [str(item) for item in design.get("forbidden_paths") or []]
            state.acceptance_checks = list(design.get("acceptance_checks") or [])
            state.integrity_anchors = {
                str(path): str(digest)
                for path, digest in (run.get("critical_file_hashes") or {}).items()
            }
            state.verification_results = list(implementation.get("verification_results") or [])

        save_runtime_state(repo_root, state, session_id=session_id)
        return state


def record_tool_observation(
    repo_root: Path,
    tool_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
    *,
    session_id: str = "default",
) -> None:
    with _runtime_state_lock:
        state = load_runtime_state(repo_root, session_id=session_id)
        sanitized_arguments = sanitize_tool_arguments(tool_name, arguments)
        if not result.ok:
            fingerprint = result.fingerprint or action_fingerprint(tool_name, arguments)
            existing = next(
                (
                    item
                    for item in state.failed_strategies
                    if item.fingerprint == fingerprint
                    and item.tool == tool_name
                    and item.error_code == result.error_code
                ),
                None,
            )
            if existing is None:
                state.failed_strategies.append(
                    FailedStrategy(
                        fingerprint=fingerprint,
                        tool=tool_name,
                        arguments=sanitized_arguments,
                        error_code=result.error_code,
                        summary=result.summary[:500],
                    )
                )
                state.failed_strategies = state.failed_strategies[-MAX_RUNTIME_ITEMS:]
            else:
                existing.occurrences += 1
                existing.summary = result.summary[:500]

        raw_command = str(arguments.get("command", ""))
        if tool_name == "bash" and classify_command(raw_command) is CommandKind.TEST:
            sanitized_command = sanitized_arguments.get("command", "[REDACTED]")
            state.recent_test_observations.append(
                TestObservation(
                    command=sanitized_command if isinstance(sanitized_command, str) else "[REDACTED]",
                    ok=result.ok,
                    error_code=result.error_code,
                    artifact_path=_portable_artifact_path(repo_root, result.artifact_path),
                )
            )
            state.recent_test_observations = state.recent_test_observations[-MAX_RUNTIME_ITEMS:]
        save_runtime_state(repo_root, state, session_id=session_id)


def _run_dir(repo_root: Path, run_id: str | None) -> Path | None:
    if run_id is None:
        return None
    # 使用工作流的权威加载器校验 run_id、仓库归属、worktree 绑定和数据模型。
    from osc_agent.workflows.contribution.state import load_run

    try:
        run = load_run(repo_root=repo_root, run_id=run_id)
    except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid contribution run: {run_id}") from exc
    return Path(run.artifacts_dir)


def _portable_artifact_path(repo_root: Path, artifact_path: str | None) -> str | None:
    if not artifact_path:
        return None
    candidate = Path(artifact_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return "[EXTERNAL_ARTIFACT]"


def _clear_run_state(state: RuntimeState) -> None:
    state.scope_exclusions = []
    state.allowed_files = []
    state.forbidden_paths = []
    state.integrity_anchors = {}
    state.acceptance_checks = []
    state.verification_results = []
    state.recent_test_observations = []
    state.failed_strategies = []


def _reset_state(question: str) -> RuntimeState:
    return RuntimeState(unresolved_questions=[question])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
