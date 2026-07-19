from __future__ import annotations

import ast
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.harness.command import run_command
from osc_agent.harness.repository_boundary import safe_repo_path
from osc_agent.harness.risk import assess_shell_risk
from osc_agent.harness.tasks import create_default_task_graph
from osc_agent.harness.todo import todo_write
from osc_agent.tools.git import git_changes, git_diff, git_status
from osc_agent.workflows.contribution.models import (
    ContributionRun,
    UnderstandingCheckpoint,
    UnderstandingDecision,
)
from osc_agent.workflows.contribution.prompts import (
    build_edit_prompt,
    build_implementation_prompt,
    build_reproduction_prompt,
    build_repair_prompt,
    build_understanding_prompt,
)
from osc_agent.workflows.contribution.scope import is_runtime_artifact as _is_runtime_artifact
from osc_agent.workflows.contribution.scope import validate_implementation_scope
from osc_agent.workflows.contribution.state import (
    _read_json,
    _require_consistent_run,
    _write_json,
    _write_metrics_report,
    _write_text,
    acquire_run_lock,
    load_run,
    save_run,
)
from osc_agent.workflows.contribution.transitions import _begin_stage, _complete_stage

def prepare_implementation_stage(
    *,
    repo_root: Path,
    run_id: str,
    github_token: str | None = None,
) -> tuple[ContributionRun, str]:
    with acquire_run_lock(repo_root=repo_root, run_id=run_id):
        return _prepare_implementation_stage_locked(
            repo_root=repo_root,
            run_id=run_id,
            github_token=github_token,
        )


def _prepare_implementation_stage_locked(
    *,
    repo_root: Path,
    run_id: str,
    github_token: str | None = None,
) -> tuple[ContributionRun, str]:
    run = load_run(repo_root=repo_root, run_id=run_id)
    existing_report = Path(run.artifacts_dir) / "03_implementation.json"
    implement_status = (run.stage_status or {}).get("implement")
    checkpoint = run.implementation_checkpoint or {}
    reproduction_failed_before_edit = (
        (checkpoint.get("reproduction") or {}).get("status") == "FAILED"
        and not checkpoint.get("edit")
    )
    if (
        run.stage == "implement"
        and implement_status == StageStatus.FAILED.value
        and existing_report.exists()
        and (not checkpoint or reproduction_failed_before_edit)
    ):
        # 生产代码尚未编辑时，允许显式重试 Understanding 或生成失败的回归测试。
        _begin_stage(run, "implement", repo_root, github_token=github_token)
        implement_status = StageStatus.RUNNING.value
    if run.stage == "implement" and implement_status != StageStatus.RUNNING.value:
        raise ValueError("implementation can resume only while the stage is RUNNING")
    if run.stage == "implement" and not existing_report.exists():
        raise ValueError("implementation cannot resume without 03_implementation.json")
    resuming = (
        run.stage == "implement"
        and implement_status == StageStatus.RUNNING.value
        and existing_report.exists()
    )
    # 实现开始后 evidence 文件本来就可能被合法修改；恢复时只校验 HEAD 与阶段产物。
    _require_consistent_run(run, repo_root, check_evidence=not resuming)
    if resuming:
        report = _read_json(run, "03_implementation.json")
        _require_checkpoint_consistency(run, report)
        design = _read_json(run, "02_design.json")
        return run, build_implementation_prompt(run, design)
    if existing_report.exists():
        raise ValueError("implementation report exists before the implementation stage starts")
    _begin_stage(run, "implement", repo_root, github_token=github_token)
    design = _read_json(run, "02_design.json")
    contribution_spec = design.get("contribution_spec") or {}
    baseline_results = (
        run_baseline_checks(repo_root, contribution_spec.get("baseline_checks") or [])
        if contribution_spec.get("task_type") == "behavior"
        else []
    )
    todo_write(
        [
            {"content": "Read the selected design and confirm implementation scope", "status": "in_progress"},
            {"content": "Implement the smallest reviewable change", "status": "pending"},
            {"content": "Run focused tests or document manual verification", "status": "pending"},
            {"content": "Summarize files, tests, risks, and PR notes", "status": "pending"},
        ],
        repo_root=repo_root,
    )
    tasks = create_default_task_graph(repo_root)
    prompt = build_implementation_prompt(run, design)
    report = {
        "selected_direction": run.selected_direction,
        "recommended": design.get("recommended"),
        "implementation_prompt": prompt,
        "created_tasks": [asdict(task) for task in tasks],
        "agent_output": "Implementation has not run yet.",
        "git_status_before": git_status(repo_root=repo_root),
        "git_status_after": "",
        "test_summary": "Not run yet.",
        "verification_results": [],
        "baseline_results": baseline_results,
        "reproduction_evidence": {},
        "reproduction_validation": {"ok": contribution_spec.get("task_type") != "behavior"},
        "requirement_coverage": [],
        "contribution_spec": contribution_spec,
        "scope_validation": {},
        "checkpoint": run.implementation_checkpoint or {},
    }
    run.implementation_checkpoint = run.implementation_checkpoint or {}
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    reproduction = contribution_spec.get("reproduction") or {}
    if (
        contribution_spec.get("task_type") == "behavior"
        and reproduction.get("mode", "existing") == "existing"
        and not all(item.get("expected_failure_matched") for item in baseline_results)
    ):
        run.final_status = RunStatus.FAILED_VALIDATION.value
        if run.metrics is not None:
            run.metrics["failure_reason"] = "pre-change failure baseline did not match"
        _complete_stage(run, "implement", success=False)
        _write_metrics_report(run)
        raise ValueError(
            "FAILED_VALIDATION: behavior change baseline did not reproduce the expected failure"
        )
    save_run(run)
    return run, prompt


def record_implementation_result(
    *,
    repo_root: Path,
    run_id: str,
    agent_output: str | None = None,
    test_summary: str | None = None,
    understanding_output: str | None = None,
    verification_output: str | None = None,
    verification_results: list[dict[str, Any]] | None = None,
) -> ContributionRun:
    with acquire_run_lock(repo_root=repo_root, run_id=run_id):
        return _record_implementation_result_locked(
            repo_root=repo_root,
            run_id=run_id,
            agent_output=agent_output,
            test_summary=test_summary,
            understanding_output=understanding_output,
            verification_output=verification_output,
            verification_results=verification_results,
        )


def _record_implementation_result_locked(
    *,
    repo_root: Path,
    run_id: str,
    agent_output: str | None = None,
    test_summary: str | None = None,
    understanding_output: str | None = None,
    verification_output: str | None = None,
    verification_results: list[dict[str, Any]] | None = None,
) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    if (run.stage_status or {}).get("implement") != StageStatus.RUNNING.value:
        raise ValueError("implement must be RUNNING before recording an implementation result")
    _require_consistent_run(run, repo_root, check_evidence=False)
    design = _read_json(run, "02_design.json")
    existing = _read_json(run, "03_implementation.json")
    _require_checkpoint_consistency(run, existing)
    report = {
        **existing,
        "selected_direction": run.selected_direction,
        "recommended": design.get("recommended"),
        "implementation_prompt": build_implementation_prompt(run, design),
        "created_tasks": existing.get("created_tasks") or [],
        "understanding_output": understanding_output or "",
        "agent_output": agent_output or "Implementation finished without captured output.",
        "verification_output": verification_output or "",
        "git_status_before": existing.get("git_status_before") or "",
        "git_status_after": git_status(repo_root=repo_root),
        "test_summary": test_summary or _infer_test_summary("\n".join([agent_output or "", verification_output or ""])),
    }
    report["verification_results"] = (
        verification_results
        if verification_results is not None
        else run_verification_commands(repo_root, design.get("tests_to_run") or [])
    )
    report["baseline_results"] = existing.get("baseline_results") or []
    report["reproduction_evidence"] = existing.get("reproduction_evidence") or {}
    report["contribution_spec"] = design.get("contribution_spec") or {}
    reproduction_mode = str((report["contribution_spec"].get("reproduction") or {}).get("mode") or "existing")
    report["reproduction_validation"] = (
        (
            _validate_frozen_reproduction(repo_root, report["reproduction_evidence"])
            if report["reproduction_evidence"]
            else {"ok": False, "mode": "generated_test", "changed_files": [], "missing_files": []}
        )
        if reproduction_mode == "generated_test"
        else {"ok": True, "mode": "existing"}
    )
    report["requirement_coverage"] = _build_requirement_coverage(
        design,
        report["verification_results"],
    )
    report["scope_validation"] = validate_implementation_scope(repo_root, design)
    if not report["scope_validation"].get("ok"):
        checkpoint = run.implementation_checkpoint or {}
        if "edit" in checkpoint:
            checkpoint["edit"]["status"] = "NEEDS_REPAIR"
        run.implementation_checkpoint = checkpoint
    report["checkpoint"] = run.implementation_checkpoint or {}
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    baseline_ok = report["contribution_spec"].get("task_type") != "behavior" or (
        bool(report["baseline_results"])
        and all(item.get("expected_failure_matched") for item in report["baseline_results"])
    )
    coverage_ok = bool(report["requirement_coverage"]) and all(
        item.get("passed") for item in report["requirement_coverage"]
    )
    passed = report["scope_validation"].get("ok", False) and baseline_ok and coverage_ok and bool(
        report["reproduction_validation"].get("ok")
    ) and all(
        item.get("exit_code") == 0 for item in report["verification_results"]
    ) and bool(report["verification_results"])
    run.final_status = RunStatus.SUCCESS.value if passed else RunStatus.FAILED_VALIDATION.value
    _update_change_metrics(run, repo_root, report)
    _complete_stage(run, "implement", success=passed)
    _write_metrics_report(run)
    return run


def execute_implementation_stage(
    *,
    repo_root: Path,
    run_id: str,
    run_step: Callable[[str, str], str],
    github_token: str | None = None,
) -> ContributionRun:
    """按 checkpoint 恢复 implementation；已完成的编辑步骤不会重复执行。"""
    with acquire_run_lock(repo_root=repo_root, run_id=run_id):
        return _execute_implementation_stage_locked(
            repo_root=repo_root,
            run_id=run_id,
            run_step=run_step,
            github_token=github_token,
        )


def _execute_implementation_stage_locked(
    *,
    repo_root: Path,
    run_id: str,
    run_step: Callable[[str, str], str],
    github_token: str | None,
) -> ContributionRun:
    run, _ = _prepare_implementation_stage_locked(
        repo_root=repo_root,
        run_id=run_id,
        github_token=github_token,
    )
    design = _read_json(run, "02_design.json")
    checkpoint = run.implementation_checkpoint or {}

    understanding_state = checkpoint.get("understanding") or {}
    understanding_output = (
        str(understanding_state.get("output") or "")
        if understanding_state.get("status") == "SUCCEEDED"
        else ""
    )
    understanding: UnderstandingCheckpoint | None = None
    if understanding_output:
        try:
            understanding = _parse_understanding_checkpoint(understanding_output, design)
        except ValueError as exc:
            _fail_implementation_validation(run, exc)
            raise
    else:
        prompt = build_understanding_prompt(run, design)
        for attempt in range(2):
            try:
                understanding_output = run_step("understanding", prompt)
            except Exception as exc:
                _fail_implementation_run(run, exc)
                raise
            try:
                understanding = _parse_understanding_checkpoint(understanding_output, design)
                break
            except ValueError as exc:
                if attempt == 0:
                    prompt = (
                        build_understanding_prompt(run, design)
                        + f"\nThe previous response failed validation: {exc}. "
                        "Return only the exact JSON object now."
                    )
                    continue
                _fail_implementation_validation(run, exc)
                raise
    if understanding is None:
        raise RuntimeError("understanding checkpoint parsing ended without a result")
    canonical_understanding = understanding.model_dump_json()
    if not understanding_state.get("output"):
        _save_implementation_checkpoint(run, "understanding", canonical_understanding)
    if understanding.decision != UnderstandingDecision.READY_TO_EDIT.value:
        run.final_status = RunStatus.BLOCKED_NEEDS_USER.value
        _complete_stage(run, "implement", success=False)
        _write_metrics_report(run)
        raise ValueError(
            "Implementation stopped at the understanding checkpoint: CONTRACT_UPDATE_REQUIRED"
        )

    spec = design.get("contribution_spec") or {}
    reproduction = spec.get("reproduction") or {}
    if spec.get("task_type") == "behavior" and reproduction.get("mode") == "generated_test":
        try:
            _prepare_generated_reproduction(
                repo_root=repo_root,
                run=run,
                design=design,
                run_step=run_step,
            )
        except Exception as exc:
            if isinstance(exc, ValueError) and "FAILED_VALIDATION" in str(exc):
                _fail_implementation_validation(run, exc)
            else:
                _fail_implementation_run(run, exc)
            raise

    reproduction_evidence = _read_json(run, "03_implementation.json").get("reproduction_evidence") or {}
    edit_prompt = build_edit_prompt(
        run,
        design,
        understanding,
        reproduction_evidence=reproduction_evidence,
    )
    max_repairs = max(1, int((run.config_snapshot or {}).get("consecutive_failure_limit", 3)))
    checkpoint = run.implementation_checkpoint or {}
    if (checkpoint.get("verification") or {}).get("status") == "FAILED" and (
        checkpoint.get("edit") or {}
    ).get("status") == "SUCCEEDED":
        checkpoint["edit"]["status"] = "NEEDS_REPAIR"
        _sync_implementation_checkpoint(run)

    while True:
        checkpoint = run.implementation_checkpoint or {}
        edit_state = checkpoint.get("edit") or {}
        needs_repair = edit_state.get("status") == "NEEDS_REPAIR"
        if needs_repair and len(checkpoint.get("repair_attempts") or []) >= max_repairs:
            exhausted_results = (checkpoint.get("verification") or {}).get("results") or []
            return _record_implementation_result_locked(
                repo_root=repo_root,
                run_id=run_id,
                understanding_output=canonical_understanding,
                agent_output=str(edit_state.get("output") or "repair limit exhausted"),
                verification_output=_verification_summary(exhausted_results),
                verification_results=exhausted_results,
            )
        agent_output = str(edit_state.get("output") or "") if edit_state.get("status") == "SUCCEEDED" else ""
        if not agent_output:
            stage_name = "repair" if needs_repair else "edit"
            prompt = (
                build_repair_prompt(
                    run,
                    design,
                    checkpoint.get("last_verification_failure") or {},
                    reproduction_evidence=reproduction_evidence,
                )
                if needs_repair
                else edit_prompt
            )
            try:
                agent_output = run_step(stage_name, prompt)
                if agent_output.strip() == UnderstandingDecision.CONTRACT_UPDATE_REQUIRED.value:
                    _save_implementation_checkpoint(run, "edit", agent_output, succeeded=False)
                    run.final_status = RunStatus.BLOCKED_NEEDS_USER.value
                    _complete_stage(run, "implement", success=False)
                    _write_metrics_report(run)
                    raise ValueError(
                        f"Implementation stopped during {stage_name}: CONTRACT_UPDATE_REQUIRED"
                    )
                checkpoint.pop("verification", None)
                _save_implementation_checkpoint(run, "edit", agent_output)
            except Exception as exc:
                if run.stage_status.get("implement") != StageStatus.RUNNING.value:
                    raise
                _fail_implementation_run(run, exc)
                raise

        reproduction_evidence = _read_json(run, "03_implementation.json").get("reproduction_evidence") or {}
        reproduction_validation = _validate_frozen_reproduction(repo_root, reproduction_evidence)
        if not reproduction_validation.get("ok"):
            exc = ValueError(
                "FAILED_VALIDATION: frozen regression test changed during production implementation"
            )
            _record_reproduction_validation(run, reproduction_validation)
            _fail_implementation_validation(run, exc)
            raise exc

        checkpoint = run.implementation_checkpoint or {}
        verification_state = checkpoint.get("verification") or {}
        verification_results = (
            verification_state.get("results") if verification_state.get("status") == "SUCCEEDED" else None
        )
        if verification_results is None:
            verification_results = run_verification_commands(repo_root, design.get("tests_to_run") or [])
            _save_implementation_checkpoint(run, "verification", "", results=verification_results)
        verification = _verification_summary(verification_results)
        if needs_repair:
            _record_repair_attempt(run, repo_root, agent_output, verification_results)

        verification_passed = bool(verification_results) and all(
            item.get("exit_code") == 0 for item in verification_results
        )
        if verification_passed or not _is_repairable_verification_failure(verification_results):
            return _record_implementation_result_locked(
                repo_root=repo_root,
                run_id=run_id,
                understanding_output=canonical_understanding,
                agent_output=agent_output,
                verification_output=verification,
                verification_results=verification_results,
            )

        _mark_edit_needs_repair(run, verification_results)
        if len((run.implementation_checkpoint or {}).get("repair_attempts") or []) >= max_repairs:
            return _record_implementation_result_locked(
                repo_root=repo_root,
                run_id=run_id,
                understanding_output=canonical_understanding,
                agent_output=agent_output,
                verification_output=verification,
                verification_results=verification_results,
            )


def implement_stage(*, repo_root: Path, run_id: str, agent_output: str | None = None) -> ContributionRun:
    with acquire_run_lock(repo_root=repo_root, run_id=run_id):
        run, _ = _prepare_implementation_stage_locked(repo_root=repo_root, run_id=run_id)
        if agent_output is None:
            return run
        return _record_implementation_result_locked(
            repo_root=repo_root,
            run_id=run_id,
            agent_output=agent_output,
        )


def record_test_waiver(*, repo_root: Path, run_id: str, reason: str) -> ContributionRun:
    with acquire_run_lock(repo_root=repo_root, run_id=run_id):
        return _record_test_waiver_locked(repo_root=repo_root, run_id=run_id, reason=reason)


def _record_test_waiver_locked(*, repo_root: Path, run_id: str, reason: str) -> ContributionRun:
    if not reason.strip():
        raise ValueError("test waiver reason is required")
    run = load_run(repo_root=repo_root, run_id=run_id)
    if (run.stage_status or {}).get("implement") != StageStatus.RUNNING.value:
        raise ValueError("implement must be RUNNING before recording a test waiver")
    _require_consistent_run(run, repo_root, check_evidence=False)
    report = _read_json(run, "03_implementation.json")
    _require_checkpoint_consistency(run, report)
    if report.get("verification_results"):
        raise ValueError("test waiver is only valid when no verification command is available")
    report["test_waiver"] = {
        "operator": getpass.getuser(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason.strip(),
    }
    design = _read_json(run, "02_design.json")
    task_type = str((design.get("contribution_spec") or {}).get("task_type") or "")
    if task_type not in {"docs", "config"}:
        raise ValueError("test waiver is only allowed for documentation or configuration tasks")
    report["contribution_spec"] = design.get("contribution_spec") or {}
    report["requirement_coverage"] = _build_requirement_coverage(
        design,
        report.get("verification_results") or [],
        manual_approved=True,
    )
    report["scope_validation"] = validate_implementation_scope(repo_root, design)
    scope_ok = bool(report["scope_validation"].get("ok"))
    coverage_ok = bool(report["requirement_coverage"]) and all(
        item.get("passed") for item in report["requirement_coverage"]
    )
    passed = scope_ok and coverage_ok
    if run.metrics is not None:
        run.metrics["human_confirmations"] = int(run.metrics.get("human_confirmations", 0)) + 1
    run.final_status = RunStatus.SUCCESS.value if passed else RunStatus.FAILED_VALIDATION.value
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    _complete_stage(run, "implement", success=passed)
    _write_metrics_report(run)
    return run


def render_implementation_report(report: dict[str, Any]) -> str:
    verification_results = report.get("verification_results") or []
    verification_table = "\n".join(
        f"- `{item.get('command')}` → exit {item.get('exit_code')} ({item.get('duration_ms')} ms)"
        for item in verification_results
    ) or "- No verification command executed."
    scope = report.get("scope_validation") or {}
    waiver = report.get("test_waiver") or {}
    baseline = report.get("baseline_results") or []
    reproduction_evidence = report.get("reproduction_evidence") or {}
    reproduction_validation = report.get("reproduction_validation") or {}
    coverage = report.get("requirement_coverage") or []
    repair_attempts = (report.get("checkpoint") or {}).get("repair_attempts") or []
    return (
        "# Implementation Report\n\n"
        f"## Selected Direction\n{report['selected_direction']}\n\n"
        f"## Recommended Approach\n{report['recommended']}\n\n"
        "## Implementation Prompt\n"
        f"```text\n{report['implementation_prompt']}\n```\n\n"
        "## Understanding\n"
        f"{report.get('understanding_output', '')}\n\n"
        "## Agent Output\n"
        f"{report['agent_output']}\n\n"
        "## Verification\n"
        f"{report.get('verification_output', '')}\n\n"
        "## Pre-change Failure Baseline\n"
        f"```json\n{json.dumps(baseline, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Generated Reproduction Evidence\n"
        f"```json\n{json.dumps(reproduction_evidence, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Frozen Test Validation\n"
        f"```json\n{json.dumps(reproduction_validation, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Requirement Coverage\n"
        f"```json\n{json.dumps(coverage, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Verification Repair Attempts\n"
        f"```json\n{json.dumps(repair_attempts, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Testing\n"
        f"{report['test_summary']}\n\n{verification_table}\n\n"
        "## Deterministic Scope Validation\n"
        f"```json\n{json.dumps(scope, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Test Waiver\n"
        f"{json.dumps(waiver, ensure_ascii=False) if waiver else 'none'}\n\n"
        "## Git Status Before\n"
        f"```text\n{report.get('git_status_before', '')}\n```\n\n"
        "## Git Status After\n"
        f"```text\n{report.get('git_status_after', '')}\n```\n"
    )


def _infer_test_summary(agent_output: str) -> str:
    lowered = agent_output.lower()
    if "pytest" in lowered or "passed" in lowered or "failed" in lowered:
        return agent_output[-2000:]
    return "No explicit test command found in captured agent output."


def _extract_code_block(text: str) -> str:
    marker = "```text"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = text.find("```", start)
    return text[start:end].strip() if end != -1 else ""


def _fail_implementation_run(run: ContributionRun, exc: Exception) -> None:
    text = str(exc)
    if "FAILED_BUDGET" in text:
        run.final_status = RunStatus.FAILED_BUDGET.value
    elif "BLOCKED_NEEDS_USER" in text:
        run.final_status = RunStatus.BLOCKED_NEEDS_USER.value
    else:
        run.final_status = RunStatus.FAILED_TOOL.value
    if run.metrics is not None:
        run.metrics["failure_reason"] = text
    _complete_stage(run, "implement", success=False)
    _write_metrics_report(run)


def _fail_implementation_validation(run: ContributionRun, exc: Exception) -> None:
    run.final_status = RunStatus.FAILED_VALIDATION.value
    if run.metrics is not None:
        run.metrics["failure_reason"] = str(exc)
    _complete_stage(run, "implement", success=False)
    _write_metrics_report(run)


def run_verification_commands(
    repo_root: Path,
    commands: list[str],
    *,
    confirm: Callable[[str], bool] | None = None,
    artifact_namespace: str = "verification",
) -> list[dict[str, Any]]:
    """评估验证命令风险并执行获准命令，同时保存审计日志。"""
    results: list[dict[str, Any]] = []
    for command in commands:
        decision = assess_shell_risk(command)
        permitted = decision.allowed or (
            decision.action == "ask" and confirm is not None and confirm(decision.reason)
        )
        if not permitted:
            results.append(
                {
                    "command": command,
                    "exit_code": -2,
                    "duration_ms": 0,
                    "artifact_path": "",
                    "permission": decision.action,
                    "error": decision.reason,
                }
            )
            continue
        result = run_command(
            command,
            repo_root=repo_root,
            timeout_seconds=300,
            artifact_namespace=artifact_namespace,
            environment=_verification_environment(repo_root),
        )
        results.append(result.model_dump(mode="json", exclude={"stdout", "stderr"}))
    return results


def _verification_environment(repo_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source_root = repo_root / "src"
    if source_root.is_dir():
        # Worktree 必须优先于虚拟环境中可能指向源仓库的 editable install。
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(source_root.resolve()), inherited) if part
        )
    return environment


def run_baseline_checks(repo_root: Path, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在修改前执行复现命令，并同时匹配退出码与稳定错误文本。"""
    results: list[dict[str, Any]] = []
    for check in checks:
        command = str(check.get("command") or "")
        result = run_verification_commands(repo_root, [command], artifact_namespace="baseline")[0]
        artifact_path = str(result.get("artifact_path") or "")
        output = ""
        if artifact_path:
            try:
                output = Path(artifact_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                output = ""
        expected_codes = [int(code) for code in check.get("expected_exit_codes") or []]
        expected_output = str(check.get("output_contains") or "")
        result["expected_exit_codes"] = expected_codes
        result["expected_output"] = expected_output
        result["expected_failure_matched"] = (
            result.get("exit_code") in expected_codes
            and bool(expected_output)
            and expected_output.casefold() in output.casefold()
        )
        results.append(result)
    return results


def _prepare_generated_reproduction(
    *,
    repo_root: Path,
    run: ContributionRun,
    design: dict[str, Any],
    run_step: Callable[[str, str], str],
) -> None:
    checkpoint = run.implementation_checkpoint or {}
    saved = checkpoint.get("reproduction") or {}
    if saved.get("status") == "SUCCEEDED":
        evidence = saved.get("evidence") or {}
        validation = _validate_frozen_reproduction(repo_root, evidence)
        if not validation.get("ok"):
            raise ValueError("FAILED_VALIDATION: frozen regression test changed before resume")
        _record_reproduction_evidence(run, evidence, validation)
        return

    reproduction = (design.get("contribution_spec") or {}).get("reproduction") or {}
    test_files = [str(path).replace("\\", "/") for path in reproduction.get("test_files") or []]
    test_paths = {path: safe_repo_path(repo_root, path) for path in test_files}
    command = str(reproduction.get("command") or "")
    prompt = build_reproduction_prompt(run, design)
    if saved.get("status") == "FAILED":
        previous = saved.get("evidence") or {}
        prompt += (
            "\nThe previous generated test failed host validation. Correct the declared test files using these "
            "structured diagnostics:\n"
            f"{json.dumps(_reproduction_failure_details(previous), ensure_ascii=False, indent=2)}"
        )
    output = run_step("reproduce", prompt)
    after = {
        change.path
        for change in git_changes(repo_root=repo_root)
        if not _is_runtime_artifact(change.path)
    }
    reproduction_changes = sorted(after)
    unauthorized = [path for path in reproduction_changes if path not in test_files]
    missing = [path for path in test_files if path not in after or not test_paths[path].is_file()]
    if not reproduction_changes or unauthorized or missing:
        raise ValueError(
            "FAILED_VALIDATION: reproduction step must change only declared test files; "
            f"unauthorized={unauthorized}, missing={missing}"
        )

    result = run_verification_commands(repo_root, [command], artifact_namespace="reproduction")[0]
    result["expected_exit_codes"] = [1]
    captured = _verification_output(result)
    result["expected_output"] = "pytest assertion failure"
    result["expected_failure_matched"] = (
        result.get("exit_code") == 1 and "failed" in captured.casefold()
    )
    semantic_binding = _analyze_reproduction_semantics(
        repo_root=repo_root,
        test_files=test_files,
        target_symbols=[str(symbol) for symbol in design.get("target_symbols") or []],
        failure_output=captured,
        requirement_ids=[
            str(item.get("id"))
            for item in (design.get("contribution_spec") or {}).get("requirements") or []
            if item.get("id")
        ],
    )
    frozen_hashes = _hash_test_files(repo_root, test_files)
    evidence = {
        "mode": "generated_test",
        "agent_output": output,
        "command": command,
        "test_files": test_files,
        "frozen_hashes": frozen_hashes,
        "baseline_result": result,
        "semantic_binding": semantic_binding,
    }
    validation = _validate_frozen_reproduction(repo_root, evidence)
    succeeded = (
        bool(result["expected_failure_matched"])
        and bool(validation.get("ok"))
        and bool(semantic_binding.get("ok"))
    )
    _save_implementation_checkpoint(
        run,
        "reproduction",
        output,
        results=[result],
        succeeded=succeeded,
        evidence=evidence,
    )
    _record_reproduction_evidence(run, evidence, validation)
    if not succeeded:
        details = _reproduction_failure_details(evidence)
        detail_text = "; ".join(details["reasons"])
        if not semantic_binding.get("matched_target_symbols"):
            raise ValueError(
                "FAILED_VALIDATION: generated regression test does not bind the failure "
                f"to an approved target symbol; {detail_text}"
            )
        raise ValueError(
            "FAILED_VALIDATION: generated regression test did not produce valid Issue-linked assertion evidence; "
            f"{detail_text}"
        )


def _hash_test_files(repo_root: Path, test_files: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in test_files:
        path = safe_repo_path(repo_root, relative)
        if path.is_file():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _reproduction_failure_details(evidence: dict[str, Any]) -> dict[str, Any]:
    baseline = evidence.get("baseline_result") or {}
    semantic = evidence.get("semantic_binding") or {}
    reasons = [str(reason) for reason in semantic.get("violations") or []]
    if not baseline.get("expected_failure_matched"):
        reasons.append(
            "pytest must exit with code 1 and report a test failure "
            f"(actual exit code: {baseline.get('exit_code')})"
        )
    if not evidence.get("frozen_hashes"):
        reasons.append("no generated test file was available to freeze")
    return {
        "reasons": reasons or ["generated reproduction evidence is incomplete"],
        "pytest_exit_code": baseline.get("exit_code"),
        "matched_target_symbols": semantic.get("matched_target_symbols") or [],
        "assertion_count": semantic.get("assertion_count", 0),
    }


def _analyze_reproduction_semantics(
    *,
    repo_root: Path,
    test_files: list[str],
    target_symbols: list[str],
    failure_output: str,
    requirement_ids: list[str],
) -> dict[str, Any]:
    assertion_count = 0
    called_symbols: set[str] = set()
    locally_defined_symbols: set[str] = set()
    syntax_errors: list[str] = []
    for relative in test_files:
        path = safe_repo_path(repo_root, relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError) as exc:
            syntax_errors.append(f"{relative}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                locally_defined_symbols.add(node.name)
            if isinstance(node, ast.Assert):
                assertion_count += 1
            elif isinstance(node, ast.Call):
                names = _call_symbol_names(node.func)
                if names:
                    called_symbols.update(names)
                    terminal = min(names, key=len)
                    if terminal in {"raises", "fail", "warns"} or terminal.casefold().startswith("assert"):
                        assertion_count += 1

    matched_targets = sorted(
        target
        for target in set(target_symbols)
        if target not in locally_defined_symbols
        and target.rsplit(".", 1)[-1] not in locally_defined_symbols
        and (
            target in called_symbols
            or target.rsplit(".", 1)[-1] in called_symbols
        )
    )
    normalized_output = failure_output.replace("\\", "/").casefold()
    failure_references_test = any(path.casefold() in normalized_output for path in test_files)
    violations: list[str] = []
    if syntax_errors:
        violations.append("test source is not valid Python")
    if assertion_count == 0:
        violations.append("no supported assertion found")
    if not matched_targets:
        violations.append("no approved target symbol is called")
    if not failure_references_test:
        violations.append("pytest failure output does not reference a declared test file")
    return {
        "ok": not violations,
        "requirement_ids": requirement_ids,
        "target_symbols": sorted(set(target_symbols)),
        "called_symbols": sorted(called_symbols),
        "locally_defined_symbols": sorted(locally_defined_symbols),
        "matched_target_symbols": matched_targets,
        "assertion_count": assertion_count,
        "failure_references_test": failure_references_test,
        "syntax_errors": syntax_errors,
        "violations": violations,
    }


def _call_symbol_names(function: ast.expr) -> set[str]:
    if isinstance(function, ast.Name):
        return {function.id}
    if isinstance(function, ast.Attribute):
        qualified = _attribute_path(function)
        return {function.attr, qualified} if qualified else {function.attr}
    return set()


def _attribute_path(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        parent = _attribute_path(expression.value)
        return f"{parent}.{expression.attr}" if parent else expression.attr
    return ""


def _validate_frozen_reproduction(repo_root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    if not evidence:
        return {"ok": True, "mode": "existing"}
    expected = evidence.get("frozen_hashes") or {}
    changed: list[str] = []
    missing: list[str] = []
    for relative, expected_hash in expected.items():
        path = safe_repo_path(repo_root, str(relative))
        if not path.is_file():
            missing.append(str(relative))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            changed.append(str(relative))
    return {
        "ok": bool(expected) and not changed and not missing,
        "mode": "generated_test",
        "changed_files": changed,
        "missing_files": missing,
    }


def _verification_output(result: dict[str, Any]) -> str:
    artifact = str(result.get("artifact_path") or "")
    if not artifact:
        return ""
    try:
        return Path(artifact).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _record_reproduction_evidence(
    run: ContributionRun,
    evidence: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    report = _read_json(run, "03_implementation.json")
    report["baseline_results"] = [evidence.get("baseline_result") or {}]
    report["reproduction_evidence"] = evidence
    report["reproduction_validation"] = validation
    report["checkpoint"] = run.implementation_checkpoint or {}
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)


def _record_reproduction_validation(run: ContributionRun, validation: dict[str, Any]) -> None:
    report = _read_json(run, "03_implementation.json")
    report["reproduction_validation"] = validation
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)


def _build_requirement_coverage(
    design: dict[str, Any],
    verification_results: list[dict[str, Any]],
    *,
    manual_approved: bool = False,
) -> list[dict[str, Any]]:
    spec = design.get("contribution_spec") or {}
    requirements = spec.get("requirements") or []
    checks = design.get("acceptance_checks") or []
    results_by_command = {
        str(item.get("command") or ""): item for item in verification_results
    }
    coverage: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_id = str(requirement.get("id") or "")
        mapped = [check for check in checks if requirement_id in (check.get("requirement_ids") or [])]
        check_results: list[dict[str, Any]] = []
        for check in mapped:
            command = str(check.get("command") or "")
            if check.get("manual_check"):
                passed = manual_approved
            else:
                passed = bool(command) and results_by_command.get(command, {}).get("exit_code") == 0
            check_results.append(
                {
                    "criterion": str(check.get("criterion") or ""),
                    "command": command,
                    "manual_check": bool(check.get("manual_check")),
                    "passed": passed,
                }
            )
        coverage.append(
            {
                "requirement_id": requirement_id,
                "requirement": str(requirement.get("text") or ""),
                "passed": bool(check_results) and all(item["passed"] for item in check_results),
                "checks": check_results,
            }
        )
    return coverage


def _require_checkpoint_consistency(run: ContributionRun, report: dict[str, Any]) -> None:
    run_checkpoint = run.implementation_checkpoint or {}
    report_checkpoint = report.get("checkpoint") or {}
    if run_checkpoint != report_checkpoint:
        raise ValueError("implementation checkpoint state diverged between run and report")


def _parse_understanding_checkpoint(
    output: str,
    design: dict[str, Any],
) -> UnderstandingCheckpoint:
    try:
        checkpoint = UnderstandingCheckpoint.model_validate_json(output)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'root'}: {error['msg']}"
            for error in exc.errors(include_input=False)[:3]
        )
        raise ValueError(
            "invalid understanding checkpoint: expected one exact JSON object matching the saved schema; "
            f"{details}"
        ) from exc
    expected_requirements = {
        str(item.get("id"))
        for item in (design.get("contribution_spec") or {}).get("requirements") or []
        if item.get("id")
    }
    expected_files = {str(path).replace("\\", "/") for path in design.get("files_to_modify") or []}
    reported_requirements = set(checkpoint.requirement_ids)
    reported_files = {path.replace("\\", "/") for path in checkpoint.files_to_modify}
    if reported_requirements != expected_requirements or reported_files != expected_files:
        raise ValueError(
            "invalid understanding checkpoint: requirements and files must exactly match the saved Contract"
        )
    return checkpoint


def _save_implementation_checkpoint(
    run: ContributionRun,
    step: str,
    output: str,
    *,
    results: list[dict[str, Any]] | None = None,
    succeeded: bool | None = None,
    evidence: dict[str, Any] | None = None,
) -> None:
    checkpoint = run.implementation_checkpoint or {}
    if succeeded is None:
        succeeded = results is None or all(item.get("exit_code") == 0 for item in results)
    value: dict[str, Any] = {
        "status": "SUCCEEDED" if succeeded else "FAILED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if output:
        value["output"] = output
    if results is not None:
        value["results"] = results
    if evidence is not None:
        value["evidence"] = evidence
    checkpoint[step] = value
    run.implementation_checkpoint = checkpoint
    report_path = Path(run.artifacts_dir) / "03_implementation.json"
    if report_path.exists():
        report = _read_json(run, "03_implementation.json")
        report["checkpoint"] = checkpoint
        _write_json(run, "03_implementation.json", report)
    save_run(run)


def _sync_implementation_checkpoint(run: ContributionRun) -> None:
    report_path = Path(run.artifacts_dir) / "03_implementation.json"
    if report_path.exists():
        report = _read_json(run, "03_implementation.json")
        report["checkpoint"] = run.implementation_checkpoint or {}
        _write_json(run, "03_implementation.json", report)
    save_run(run)


def _mark_edit_needs_repair(run: ContributionRun, results: list[dict[str, Any]]) -> None:
    checkpoint = run.implementation_checkpoint or {}
    edit = checkpoint.get("edit") or {}
    edit["status"] = "NEEDS_REPAIR"
    checkpoint["edit"] = edit
    failure = {
        "failure_number": len(checkpoint.get("verification_failures") or []) + 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "summary": _verification_summary(results),
    }
    checkpoint.setdefault("verification_failures", []).append(failure)
    checkpoint["last_verification_failure"] = failure
    run.implementation_checkpoint = checkpoint
    _sync_implementation_checkpoint(run)


def _record_repair_attempt(
    run: ContributionRun,
    repo_root: Path,
    output: str,
    results: list[dict[str, Any]],
) -> None:
    checkpoint = run.implementation_checkpoint or {}
    passed = bool(results) and all(item.get("exit_code") == 0 for item in results)
    attempts = checkpoint.setdefault("repair_attempts", [])
    attempts.append(
        {
            "attempt": len(attempts) + 1,
            "status": "SUCCEEDED" if passed else "FAILED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "agent_output": output,
            "verification_results": results,
            "diff": git_diff(repo_root=repo_root)[:12000],
        }
    )
    run.implementation_checkpoint = checkpoint
    _sync_implementation_checkpoint(run)


def _is_repairable_verification_failure(results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    has_failure = False
    for item in results:
        code = int(item.get("exit_code", -2))
        if code < 0:
            return False
        if code == 0:
            continue
        has_failure = True
        command_tokens = str(item.get("command") or "").casefold().split()
        if "pytest" in command_tokens and code != 1:
            return False
    return has_failure


def _verification_summary(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No verification command configured."
    passed = sum(1 for item in results if item.get("exit_code") == 0)
    failed = len(results) - passed
    return f"Controlled verification completed: {passed} passed, {failed} failed."


def _update_change_metrics(run: ContributionRun, repo_root: Path, report: dict[str, Any]) -> None:
    if run.metrics is None:
        return
    scope = report.get("scope_validation") or {}
    verification = report.get("verification_results") or []
    run.metrics.update(
        {
            "changed_files": len(scope.get("changed_files") or []),
            "added_lines": int(scope.get("added_lines") or 0),
            "deleted_lines": int(scope.get("deleted_lines") or 0),
            "test_commands": len(verification),
            "test_failures": sum(1 for item in verification if item.get("exit_code") != 0),
            "repair_attempts": len((report.get("checkpoint") or {}).get("repair_attempts") or []),
            "verification_failures": len(
                (report.get("checkpoint") or {}).get("verification_failures") or []
            ),
            "final_status": run.final_status,
        }
    )
