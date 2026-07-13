from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Any

from osc_agent.harness.contracts import ToolResult
from osc_agent.tools.git import git_changed_files


RUNTIME_STATE_SCHEMA_VERSION = 1
_runtime_state_lock = threading.RLock()


@dataclass
class RuntimeState:
    schema_version: int = RUNTIME_STATE_SCHEMA_VERSION
    current_goal: str = ""
    user_constraints: list[str] = field(default_factory=list)
    allowed_files: list[str] = field(default_factory=list)
    forbidden_paths: list[str] = field(default_factory=list)
    verified_facts: list[dict[str, Any]] = field(default_factory=list)
    current_plan: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    failed_strategies: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)


def runtime_state_path(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "runtime_state.json"


def load_runtime_state(repo_root: Path) -> RuntimeState:
    with _runtime_state_lock:
        path = runtime_state_path(repo_root)
        if not path.exists():
            return RuntimeState()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return RuntimeState(unresolved_questions=["runtime state could not be parsed and was reset"])
        if data.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION:
            return RuntimeState(unresolved_questions=["runtime state schema is incompatible and was reset"])
        return RuntimeState(**data)


def save_runtime_state(repo_root: Path, state: RuntimeState) -> None:
    with _runtime_state_lock:
        path = runtime_state_path(repo_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            temp.write_text(
                json.dumps(asdict(state), ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()


def refresh_runtime_state(repo_root: Path, current_task: str) -> RuntimeState:
    with _runtime_state_lock:
        state = load_runtime_state(repo_root)
        if current_task and not current_task.startswith("tool_result "):
            state.current_goal = current_task

        run_dir = _latest_run_dir(repo_root)
        if run_dir is not None:
            run = _read_json(run_dir / "run.json")
            design = _read_json(run_dir / "02_design.json")
            implementation = _read_json(run_dir / "03_implementation.json")
            if design:
                state.user_constraints = [str(item) for item in design.get("out_of_scope") or []]
                state.allowed_files = [str(item) for item in design.get("allowed_files") or []]
                state.forbidden_paths = [str(item) for item in design.get("forbidden_paths") or []]
                state.current_plan = list(design.get("acceptance_checks") or [])
            if run:
                state.verified_facts = [
                    {"file": path, "content_hash": digest}
                    for path, digest in (run.get("critical_file_hashes") or {}).items()
                ]
            if implementation:
                state.test_results = list(implementation.get("verification_results") or [])

        state.modified_files = [
            path for path in git_changed_files(repo_root=repo_root) if not path.startswith(".osc_agent/")
        ]
        save_runtime_state(repo_root, state)
        return state


def record_tool_observation(
    repo_root: Path,
    tool_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
) -> None:
    with _runtime_state_lock:
        state = load_runtime_state(repo_root)
        if not result.ok:
            observation = {
                "tool": tool_name,
                "arguments": arguments,
                "error_code": result.error_code,
                "summary": result.summary[:500],
            }
            if observation not in state.failed_strategies:
                state.failed_strategies.append(observation)
                state.failed_strategies = state.failed_strategies[-20:]
        if tool_name == "bash" and _looks_like_test(str(arguments.get("command", ""))):
            state.test_results.append(
                {
                    "command": str(arguments.get("command", "")),
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "artifact_path": result.artifact_path,
                }
            )
            state.test_results = state.test_results[-20:]
        save_runtime_state(repo_root, state)


def _latest_run_dir(repo_root: Path) -> Path | None:
    root = repo_root / ".osc_agent" / "contribution_runs"
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "run.json").exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _looks_like_test(command: str) -> bool:
    lowered = command.lower()
    return any(marker in lowered for marker in ("pytest", "unittest", "npm test", "cargo test"))
