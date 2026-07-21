from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock, Timeout

from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.harness.repository_boundary import safe_repo_path
from osc_agent.tools.git import git_changes, git_common_dir, git_head, git_toplevel
from osc_agent.workflows.contribution.models import ContributionRun, STATE_SCHEMA_VERSION, STAGE_ORDER

STATE_LOCK_TIMEOUT_SECONDS = 10

def create_run(*, repo_root: Path, repo_url: str, settings: Any | None = None) -> ContributionRun:
    base_commit = _require_clean_source_repository(repo_root)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"
    run = ContributionRun(
        run_id=run_id,
        repo_root=str(repo_root.resolve()),
        repo_url=repo_url,
        stage="discover",
        selected_direction=None,
        artifacts_dir=str(_runs_dir(repo_root) / run_id),
        base_commit_sha=base_commit,
        issue_snapshot_at=datetime.now(timezone.utc).isoformat(),
        config_snapshot=_settings_snapshot(settings),
        stage_status={stage: StageStatus.PENDING.value for stage in STAGE_ORDER},
        stage_hashes={},
        critical_file_hashes={},
        metrics={
            "stages": {},
            "human_confirmations": 0,
            "human_modifications": 0,
            "trace_start_line": _trace_line_count(repo_root),
        },
        implementation_checkpoint={},
    )
    save_run(run)
    return run


def load_run(*, repo_root: Path, run_id: str) -> ContributionRun:
    state_root = _state_repo_root(repo_root, run_id)
    run_dir = _validated_run_dir(state_root, run_id)
    path = run_dir / "run.json"
    if not path.exists():
        raise ValueError(f"contribution run not found: {run_id}")
    data = _read_json_object(path, label="contribution run state")
    if data.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported contribution run schema; expected {STATE_SCHEMA_VERSION}. Start a new run."
        )
    if data.get("run_id") != run_id:
        raise ValueError("contribution run id does not match its directory")
    if Path(str(data.get("repo_root", ""))).resolve() != state_root.resolve():
        raise ValueError("contribution run belongs to a different repository")
    requested_root = repo_root.resolve()
    if state_root != requested_root:
        if Path(str(data.get("worktree_root", ""))).resolve() != requested_root:
            raise ValueError("worktree is not bound by the authoritative contribution run")
        _require_shared_git_common_dir(state_root, requested_root)
    if Path(str(data.get("artifacts_dir", ""))).resolve() != run_dir:
        raise ValueError("contribution run artifacts path is outside the expected run directory")
    return ContributionRun.model_validate(data)


def save_run(run: ContributionRun) -> None:
    repo_root = Path(run.repo_root).resolve()
    artifacts = _validated_run_dir(repo_root, run.run_id)
    if Path(run.artifacts_dir).resolve() != artifacts:
        raise ValueError("contribution run artifacts path is outside the expected run directory")
    artifacts.mkdir(parents=True, exist_ok=True)
    state_path = artifacts / "run.json"
    lock = FileLock(artifacts / "state.lock", timeout=STATE_LOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            if state_path.exists():
                persisted = _read_json_object(state_path, label="contribution run state")
                persisted_revision = _revision_from_payload(persisted)
                if persisted_revision != run.revision:
                    raise ValueError(
                        f"STALE_RUN: run revision changed from {run.revision} to {persisted_revision}"
                    )
            next_revision = run.revision + 1
            payload = run.model_dump(mode="json", by_alias=True)
            payload["revision"] = next_revision
            _write_raw_json(state_path, payload)
            # 只有磁盘替换成功后才提交内存 revision，避免对象与磁盘状态分叉。
            run.revision = next_revision
    except Timeout as exc:
        raise ValueError(f"contribution run state is busy: {run.run_id}") from exc


@contextmanager
def acquire_run_lock(*, repo_root: Path, run_id: str, timeout: float = 0) -> Iterator[None]:
    """阻止两个进程同时推进同一个贡献 run。"""
    lock_path = _validated_run_dir(_state_repo_root(repo_root, run_id), run_id) / "run.lock"
    lock = FileLock(lock_path, timeout=timeout)
    try:
        with lock:
            yield
    except Timeout as exc:
        raise ValueError(f"BLOCKED_NEEDS_USER: contribution run {run_id} is already executing") from exc


def _runs_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "contribution_runs"


def bind_run_worktree(*, repo_root: Path, run_id: str, worktree_root: Path) -> ContributionRun:
    """将 worktree 绑定到源仓库中的权威 Run；worktree 不复制状态或产物。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    worktree = worktree_root.resolve()
    if not worktree.is_dir():
        raise ValueError("contribution worktree does not exist")
    _require_shared_git_common_dir(Path(run.repo_root), worktree)
    run.worktree_root = str(worktree)
    save_run(run)
    _write_raw_json(
        worktree / ".osc_agent" / "contribution_run_ref.json",
        {"run_id": run_id, "state_repo_root": str(Path(run.repo_root).resolve())},
    )
    return run


def _state_repo_root(repo_root: Path, run_id: str) -> Path:
    requested = repo_root.resolve()
    if _validated_run_dir(requested, run_id).joinpath("run.json").exists():
        return requested
    reference = requested / ".osc_agent" / "contribution_run_ref.json"
    if not reference.exists():
        return requested
    try:
        data = _read_json_object(reference, label="contribution run reference")
    except ValueError as exc:
        raise ValueError(f"invalid contribution run reference: {exc}") from exc
    if data.get("run_id") != run_id:
        raise ValueError("worktree is bound to a different contribution run")
    state_root = Path(str(data.get("state_repo_root", ""))).resolve()
    if not state_root.is_dir():
        raise ValueError("contribution run source repository is unavailable")
    return state_root


def _validated_run_dir(repo_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, numbers, underscores, or hyphens")
    return safe_repo_path(repo_root, f".osc_agent/contribution_runs/{run_id}")


def _write_json(run: ContributionRun, name: str, value: dict[str, Any]) -> None:
    """写入 JSON artifact 并更新内存 hash；调用方负责随后 save_run()。"""
    _write_raw_json(_artifact_path(run, name), value)
    if run.stage_hashes is not None:
        run.stage_hashes[name] = _content_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _write_raw_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(path, content)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: JSON root must be an object")
    return value


def _revision_from_payload(payload: dict[str, Any]) -> int:
    value = payload.get("revision", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid contribution run state: revision must be a non-negative integer")
    return value


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        # 清理属于尽力而为，不能覆盖写入或 replace 的原始异常。
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(run: ContributionRun, name: str) -> dict[str, Any]:
    path = _artifact_path(run, name)
    if not path.exists():
        raise ValueError(f"required artifact missing: {name}")
    return _read_json_object(path, label=f"artifact {name}")


def _read_text(run: ContributionRun, name: str, default: str = "") -> str:
    path = _artifact_path(run, name)
    return path.read_text(encoding="utf-8") if path.exists() else default


def _write_text(run: ContributionRun, name: str, value: str) -> None:
    path = _artifact_path(run, name)
    _atomic_write_bytes(path, (value.rstrip() + "\n").encode("utf-8"))


def _artifact_path(run: ContributionRun, name: str) -> Path:
    return safe_repo_path(Path(run.artifacts_dir), name)


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_clean_source_repository(repo_root: Path) -> str:
    head = git_head(repo_root=repo_root).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        raise ValueError("target repository must be a Git repository with at least one commit")
    external = [
        change.path
        for change in git_changes(repo_root=repo_root)
        if not change.path.startswith(".osc_agent/")
    ]
    if external:
        raise ValueError("target repository has uncommitted changes; commit or remove them before discovery")
    return head


def _settings_snapshot(settings: Any | None) -> dict[str, int]:
    defaults = {
        "max_agent_rounds": 30,
        "max_total_tokens": 200_000,
        "agent_deadline_seconds": 1_800,
        "repeat_action_limit": 3,
        "consecutive_failure_limit": 3,
        "no_progress_limit": 6,
        "max_changed_files": 5,
        "max_diff_lines": 400,
    }
    if settings is None:
        return defaults
    return {name: int(getattr(settings, name, value)) for name, value in defaults.items()}


def _require_consistent_run(run: ContributionRun, repo_root: Path, *, check_evidence: bool = True) -> None:
    head = git_head(repo_root=repo_root).strip()
    if head != run.base_commit_sha:
        _raise_stale_run(run, f"repository HEAD changed from {run.base_commit_sha} to {head}")
    for name, expected in (run.stage_hashes or {}).items():
        try:
            path = _artifact_path(run, name)
        except ValueError:
            _raise_stale_run(run, f"artifact path escapes run directory: {name}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        actual = _content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        if payload is None or actual != expected:
            _raise_stale_run(run, f"stage artifact changed: {name}")
    if not check_evidence:
        return
    for relative, expected in (run.critical_file_hashes or {}).items():
        try:
            path = safe_repo_path(repo_root, relative)
        except ValueError:
            _raise_stale_run(run, f"evidence path escapes repository: {relative}")
        if not path.is_file() or _file_hash(path) != expected:
            _raise_stale_run(run, f"evidence file changed: {relative}")


def _raise_stale_run(run: ContributionRun, reason: str) -> None:
    message = f"STALE_RUN: {reason}"
    run.final_status = RunStatus.STALE_RUN.value
    try:
        save_run(run)
    except Exception as exc:
        raise ValueError(f"{message}; failed to persist stale status: {exc}") from exc
    raise ValueError(message)


def _evidence_file_hashes(repo_root: Path, evidence_pack: dict[str, Any]) -> dict[str, str]:
    files: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            candidate = value.get("file")
            if isinstance(candidate, str):
                files.add(candidate.replace("\\", "/"))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(evidence_pack)
    hashes: dict[str, str] = {}
    for relative in sorted(files):
        path = safe_repo_path(repo_root, relative)
        if path.is_file():
            hashes[relative] = _file_hash(path)
    return hashes


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_shared_git_common_dir(source_root: Path, worktree_root: Path) -> None:
    if _resolved_git_toplevel(source_root) != source_root.resolve():
        raise ValueError("contribution source path is not a Git worktree root")
    if _resolved_git_toplevel(worktree_root) != worktree_root.resolve():
        raise ValueError("contribution worktree path is not a Git worktree root")
    source_common = _resolved_git_common_dir(source_root)
    worktree_common = _resolved_git_common_dir(worktree_root)
    if source_common != worktree_common:
        raise ValueError("worktree does not belong to the contribution source repository")


def _resolved_git_common_dir(repo_root: Path) -> Path:
    output = git_common_dir(repo_root=repo_root)
    if output.startswith("Error:"):
        raise ValueError(f"unable to resolve Git common directory: {output}")
    path = Path(output)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolved_git_toplevel(repo_root: Path) -> Path:
    output = git_toplevel(repo_root=repo_root)
    if output.startswith("Error:"):
        raise ValueError(f"unable to resolve Git worktree root: {output}")
    return Path(output).resolve()


def _write_metrics_report(run: ContributionRun) -> None:
    if run.metrics is not None:
        run.metrics.update(_aggregate_trace_metrics(Path(run.repo_root), int(run.metrics.get("trace_start_line", 0))))
        save_run(run)
    metrics = {**(run.metrics or {}), "final_status": run.final_status, "run_id": run.run_id}
    _write_raw_json(Path(run.artifacts_dir) / "metrics.json", metrics)
    stages = metrics.get("stages") or {}
    rows = "\n".join(
        f"| {name} | {value.get('duration_ms', 0)} |"
        for name, value in sorted(stages.items())
    ) or "| - | 0 |"
    _write_text(
        run,
        "metrics.md",
        "# Run Metrics\n\n"
        f"- Final status: {run.final_status or 'IN_PROGRESS'}\n"
        f"- Changed files: {metrics.get('changed_files', 0)}\n"
        f"- Diff lines: {metrics.get('added_lines', 0) + metrics.get('deleted_lines', 0)}\n"
        f"- Test commands: {metrics.get('test_commands', 0)}\n\n"
        "| Stage | Duration (ms) |\n|---|---:|\n"
        f"{rows}\n",
    )


def _trace_line_count(repo_root: Path) -> int:
    path = repo_root / ".osc_agent" / "traces" / "session.jsonl"
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _aggregate_trace_metrics(repo_root: Path, start_line: int) -> dict[str, int]:
    path = repo_root / ".osc_agent" / "traces" / "session.jsonl"
    totals = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": 0,
        "tool_failures": 0,
        "model_retries": 0,
    }
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines()[start_line:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "agent_run_finished":
            metrics = event.get("metrics") or {}
            for key in ("model_calls", "input_tokens", "output_tokens", "tool_calls", "tool_failures"):
                totals[key] += int(metrics.get(key) or 0)
            totals["model_retries"] += int(metrics.get("retries") or 0)
        elif event.get("event") == "stage_model_usage":
            totals["model_calls"] += 1
            totals["input_tokens"] += int(event.get("input_tokens") or 0)
            totals["output_tokens"] += int(event.get("output_tokens") or 0)
    return totals
