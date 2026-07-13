from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.harness.permissions import safe_repo_path
from osc_agent.tools.git import git_head, git_status
from osc_agent.workflows.contribution.models import ContributionRun, STATE_SCHEMA_VERSION, STAGES

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
        stage_status={stage: StageStatus.PENDING.value for stage in sorted(STAGES)},
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
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported contribution run schema; expected {STATE_SCHEMA_VERSION}. Start a new run."
        )
    if data.get("run_id") != run_id:
        raise ValueError("contribution run id does not match its directory")
    if Path(str(data.get("repo_root", ""))).resolve() != state_root.resolve():
        raise ValueError("contribution run belongs to a different repository")
    if state_root != repo_root.resolve() and Path(str(data.get("worktree_root", ""))).resolve() != repo_root.resolve():
        raise ValueError("worktree is not bound by the authoritative contribution run")
    if Path(str(data.get("artifacts_dir", ""))).resolve() != run_dir:
        raise ValueError("contribution run artifacts path is outside the expected run directory")
    return ContributionRun(**data)


def save_run(run: ContributionRun) -> None:
    repo_root = Path(run.repo_root).resolve()
    artifacts = _validated_run_dir(repo_root, run.run_id)
    if Path(run.artifacts_dir).resolve() != artifacts:
        raise ValueError("contribution run artifacts path is outside the expected run directory")
    artifacts.mkdir(parents=True, exist_ok=True)
    _write_raw_json(artifacts / "run.json", asdict(run))


def _runs_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "contribution_runs"


def bind_run_worktree(*, repo_root: Path, run_id: str, worktree_root: Path) -> ContributionRun:
    """将 worktree 绑定到源仓库中的权威 Run；worktree 不复制状态或产物。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    worktree = worktree_root.resolve()
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
        data = json.loads(reference.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
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
    _write_raw_json(Path(run.artifacts_dir) / name, value)
    if run.stage_hashes is not None:
        run.stage_hashes[name] = _content_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _write_raw_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_json(run: ContributionRun, name: str) -> dict[str, Any]:
    path = Path(run.artifacts_dir, name)
    if not path.exists():
        raise ValueError(f"required artifact missing: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(run: ContributionRun, name: str, default: str = "") -> str:
    path = Path(run.artifacts_dir) / name
    return path.read_text(encoding="utf-8") if path.exists() else default


def _write_text(run: ContributionRun, name: str, value: str) -> None:
    path = Path(run.artifacts_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_clean_source_repository(repo_root: Path) -> str:
    head = git_head(repo_root=repo_root).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
        raise ValueError("target repository must be a Git repository with at least one commit")
    status = git_status(repo_root=repo_root)
    external = [line for line in status.splitlines() if ".osc_agent/" not in line.replace("\\", "/")]
    if status != "(no output)" and any(line.strip() for line in external):
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
        run.final_status = RunStatus.STALE_RUN.value
        save_run(run)
        raise ValueError(f"STALE_RUN: repository HEAD changed from {run.base_commit_sha} to {head}")
    for name, expected in (run.stage_hashes or {}).items():
        path = Path(run.artifacts_dir) / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        actual = _content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        if payload is None or actual != expected:
            run.final_status = RunStatus.STALE_RUN.value
            save_run(run)
            raise ValueError(f"STALE_RUN: stage artifact changed: {name}")
    if not check_evidence:
        return
    for relative, expected in (run.critical_file_hashes or {}).items():
        path = repo_root / relative
        if not path.exists() or _content_hash(path.read_text(encoding="utf-8", errors="replace")) != expected:
            run.final_status = RunStatus.STALE_RUN.value
            save_run(run)
            raise ValueError(f"STALE_RUN: evidence file changed: {relative}")


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
        path = repo_root / relative
        if path.is_file():
            hashes[relative] = _content_hash(path.read_text(encoding="utf-8", errors="replace"))
    return hashes


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

