from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.harness.repository_boundary import safe_repo_path
from osc_agent.tools.git import git_changes, git_diff, git_head
from osc_agent.workflows.contribution.agents import run_pr_draft_generation
from osc_agent.workflows.contribution.gates import gate_implementation
from osc_agent.workflows.contribution.models import (
    ContributionRun,
    PRDraftArtifact,
    PRDraftNarrative,
)
from osc_agent.workflows.contribution.scope import is_runtime_artifact
from osc_agent.workflows.contribution.state import (
    _read_json,
    _require_consistent_run,
    _write_json,
    _write_metrics_report,
    _write_text,
    acquire_run_lock,
    load_run,
)
from osc_agent.workflows.contribution.transitions import _begin_stage, _complete_stage


def draft_pr_stage(
    *,
    repo_root: Path,
    run_id: str,
    client: Any,
    settings: Any,
) -> ContributionRun:
    _require_llm(client, settings)
    with acquire_run_lock(repo_root=repo_root, run_id=run_id):
        run = load_run(repo_root=repo_root, run_id=run_id)
        _require_consistent_run(run, repo_root, check_evidence=False)
        _begin_stage(run, "draft_pr", repo_root)
        try:
            artifact = _build_pr_draft_artifact(
                repo_root=repo_root,
                run_id=run_id,
                client=client,
                settings=settings,
            )
            _write_json(run, "04_pr_draft.json", artifact.model_dump(mode="json"))
            _write_text(run, "04_pr_draft.md", render_pr_draft(artifact))
            _complete_stage(run, "draft_pr", success=True)
        except Exception as exc:
            run.final_status = (
                RunStatus.FAILED_VALIDATION.value
                if isinstance(exc, ValueError)
                else RunStatus.FAILED_TOOL.value
            )
            if run.metrics is not None:
                run.metrics["failure_reason"] = str(exc)[:1000]
            if (run.stage_status or {}).get("draft_pr") == StageStatus.RUNNING.value:
                _complete_stage(run, "draft_pr", success=False)
            _write_metrics_report(run)
            raise
        _write_metrics_report(run)
        return run


def build_workflow_pr_draft(
    *,
    repo_root: Path,
    run_id: str,
    client: Any,
    settings: Any,
) -> str:
    """基于权威 Run 产物和当前实现 worktree 生成 PR 草稿。"""
    _require_llm(client, settings)
    artifact = _build_pr_draft_artifact(
        repo_root=repo_root,
        run_id=run_id,
        client=client,
        settings=settings,
    )
    return render_pr_draft(artifact)


def _build_pr_draft_artifact(
    *,
    repo_root: Path,
    run_id: str,
    client: Any,
    settings: Any,
) -> PRDraftArtifact:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root, check_evidence=False)
    if (run.stage_status or {}).get("implement") != StageStatus.SUCCEEDED.value:
        raise ValueError("implement must be SUCCEEDED before drafting a pull request")

    gate = gate_implementation(Path(run.artifacts_dir), repo_root)
    if not gate.passed:
        raise ValueError(f"implementation gate failed: {gate.reason}")

    design = _read_json(run, "02_design.json")
    implementation = _read_json(run, "03_implementation.json")
    diff = git_diff(repo_root=repo_root)
    if diff.startswith("Error:"):
        raise RuntimeError(diff)
    changed_files = sorted(
        {
            change.path
            for change in git_changes(repo_root=repo_root)
            if not is_runtime_artifact(change.path)
        }
    )
    head = git_head(repo_root=repo_root).strip()
    diff_hash = _worktree_diff_hash(repo_root, head, diff, changed_files)

    # 已完成的草稿只能读取，且必须仍对应同一个 worktree 快照。
    if (run.stage_status or {}).get("draft_pr") == StageStatus.SUCCEEDED.value:
        existing = PRDraftArtifact.model_validate(_read_json(run, "04_pr_draft.json"))
        if existing.diff_hash != diff_hash:
            raise ValueError("STALE_RUN: PR draft no longer matches the current worktree")
        return existing

    narrative = PRDraftNarrative.model_validate(
        run_pr_draft_generation(
            client,
            settings,
            {
                "selected_direction": design.get("selected_direction") or run.selected_direction,
                "design_summary": design,
                "implementation_report": implementation,
                "git_diff": diff,
                "changed_files": changed_files,
            },
            repo_root=repo_root,
        )
    )
    implementation_hash = str((run.stage_hashes or {}).get("03_implementation.json") or "")
    return PRDraftArtifact(
        **narrative.model_dump(),
        run_id=run.run_id,
        run_revision=run.revision,
        head_sha=head,
        diff_hash=diff_hash,
        implementation_artifact_hash=implementation_hash,
        generated_at=datetime.now(timezone.utc).isoformat(),
        changed_files=changed_files,
        changes=[f"Updated `{path}`." for path in changed_files],
        testing=_testing_facts(implementation),
    )


def render_pr_draft(draft: PRDraftArtifact) -> str:
    changes = "\n".join(f"- {item}" for item in draft.changes)
    testing = "\n".join(f"- {item}" for item in draft.testing)
    notes = "\n".join(f"- {item}" for item in draft.reviewer_notes)
    return (
        "Title:\n"
        f"`{draft.title}`\n\n"
        "**Problem**\n"
        f"{draft.problem}\n\n"
        "**Solution**\n"
        f"{draft.solution}\n\n"
        "**Changes**\n"
        f"{changes}\n\n"
        "**Testing**\n"
        f"{testing}\n\n"
        "**Notes for Reviewer**\n"
        f"{notes}"
    )


def _testing_facts(implementation: dict[str, Any]) -> list[str]:
    results = implementation.get("verification_results") or []
    if results:
        return [
            f"`{item.get('command')}` → exit {item.get('exit_code')} ({item.get('duration_ms', 0)} ms)"
            for item in results
        ]
    waiver = implementation.get("test_waiver") or {}
    reason = str(waiver.get("reason") or "").strip()
    if reason:
        return [f"Test waiver recorded: {reason}"]
    raise ValueError("implementation artifact contains no verification result or test waiver")


def _worktree_diff_hash(repo_root: Path, head: str, diff: str, changed_files: list[str]) -> str:
    files: list[dict[str, str | None]] = []
    for relative in changed_files:
        path = safe_repo_path(repo_root, relative)
        content_hash = None
        if path.is_file() and not path.is_symlink():
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "content_hash": content_hash})
    payload = json.dumps(
        {"head": head, "diff": diff, "files": files},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_llm(client: Any, settings: Any) -> None:
    if client is None or settings is None:
        raise ValueError("LLM client and settings are required for Draft PR generation")
