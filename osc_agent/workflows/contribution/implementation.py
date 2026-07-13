from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.harness.permissions import check_shell_command
from osc_agent.harness.tasks import create_default_task_graph
from osc_agent.harness.todo import todo_write
from osc_agent.tools.git import git_changed_files, git_diff_numstat, git_status
from osc_agent.workflows.contribution.design import render_design
from osc_agent.workflows.contribution.models import ContributionRun, DEFAULT_FORBIDDEN_PATHS
from osc_agent.workflows.contribution.state import (
    _content_hash,
    _read_json,
    _require_consistent_run,
    _write_json,
    _write_metrics_report,
    _write_text,
    load_run,
    save_run,
)
from osc_agent.workflows.contribution.transitions import _begin_stage, _complete_stage

def prepare_implementation_stage(*, repo_root: Path, run_id: str) -> tuple[ContributionRun, str]:
    run = load_run(repo_root=repo_root, run_id=run_id)
    existing_report = Path(run.artifacts_dir) / "03_implementation.json"
    resuming = run.stage == "implement" and existing_report.exists()
    # 实现开始后 evidence 文件本来就可能被合法修改；恢复时只校验 HEAD 与阶段产物。
    _require_consistent_run(run, repo_root, check_evidence=not resuming)
    if (
        run.stage == "implement"
        and (run.stage_status or {}).get("implement") == StageStatus.RUNNING.value
        and existing_report.exists()
    ):
        design = _read_json(run, "02_design.json")
        return run, build_implementation_prompt(run, design)
    _begin_stage(run, "implement", repo_root)
    design = _read_json(run, "02_design.json")
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
        "scope_validation": {},
        "checkpoint": run.implementation_checkpoint or {},
    }
    run.implementation_checkpoint = run.implementation_checkpoint or {}
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
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
    run = load_run(repo_root=repo_root, run_id=run_id)
    if (run.stage_status or {}).get("implement") != StageStatus.RUNNING.value:
        run, _ = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    existing = _read_json(run, "03_implementation.json")
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
    report["scope_validation"] = validate_implementation_scope(repo_root, design)
    if not report["scope_validation"].get("ok"):
        checkpoint = run.implementation_checkpoint or {}
        if "edit" in checkpoint:
            checkpoint["edit"]["status"] = "NEEDS_REPAIR"
        run.implementation_checkpoint = checkpoint
    report["checkpoint"] = run.implementation_checkpoint or {}
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    passed = report["scope_validation"].get("ok", False) and all(
        item.get("exit_code") == 0 for item in report["verification_results"]
    ) and bool(report["verification_results"])
    run.final_status = RunStatus.SUCCESS.value if passed else RunStatus.FAILED_VALIDATION.value
    _complete_stage(run, "implement", success=passed)
    _update_change_metrics(run, repo_root, report)
    save_run(run)
    _write_metrics_report(run)
    return run


def execute_implementation_stage(
    *,
    repo_root: Path,
    run_id: str,
    run_step: Callable[[str, str], str],
) -> ContributionRun:
    """按 checkpoint 恢复 implementation；已完成的编辑步骤不会重复执行。"""
    run, fallback_prompt = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    checkpoint = run.implementation_checkpoint or {}

    understanding_state = checkpoint.get("understanding") or {}
    understanding = str(understanding_state.get("output") or "") if understanding_state.get("status") == "SUCCEEDED" else ""
    if not understanding:
        try:
            understanding = run_step("understanding", build_understanding_prompt(run, design))
            _save_implementation_checkpoint(run, "understanding", understanding)
        except Exception as exc:
            _fail_implementation_run(run, exc)
            raise
    if "READY_TO_EDIT" not in understanding:
        run.final_status = RunStatus.BLOCKED_NEEDS_USER.value
        _complete_stage(run, "implement", success=False)
        save_run(run)
        _write_metrics_report(run)
        raise ValueError(
            "Implementation stopped at the understanding checkpoint: "
            "the agent did not confirm READY_TO_EDIT."
        )

    edit_prompt = build_edit_prompt(run, design, understanding) or fallback_prompt
    edit_state = checkpoint.get("edit") or {}
    agent_output = str(edit_state.get("output") or "") if edit_state.get("status") == "SUCCEEDED" else ""
    if not agent_output:
        try:
            agent_output = run_step("edit", edit_prompt)
            _save_implementation_checkpoint(run, "edit", agent_output)
        except Exception as exc:
            _fail_implementation_run(run, exc)
            raise
    verification_state = checkpoint.get("verification") or {}
    verification_results = (
        verification_state.get("results") if verification_state.get("status") == "SUCCEEDED" else None
    )
    if verification_results is None:
        verification_results = run_verification_commands(repo_root, design.get("tests_to_run") or [])
        _save_implementation_checkpoint(run, "verification", "", results=verification_results)
    verification = _verification_summary(verification_results)
    return record_implementation_result(
        repo_root=repo_root,
        run_id=run_id,
        understanding_output=understanding,
        agent_output=agent_output,
        verification_output=verification,
        verification_results=verification_results,
    )


def implement_stage(*, repo_root: Path, run_id: str, agent_output: str | None = None) -> ContributionRun:
    if agent_output is None:
        run, _ = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
        return run
    return record_implementation_result(repo_root=repo_root, run_id=run_id, agent_output=agent_output)


def build_understanding_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    return (
        "OpenSourcePR implementation step 3a: understand the task before editing.\n"
        "Do not modify files in this step.\n"
        f"Selected direction: {run.selected_direction}\n"
        f"Files to inspect: {', '.join(design.get('files_to_modify') or ['not specified'])}\n"
        "Read the referenced files, summarize the implementation boundary, and explicitly say READY_TO_EDIT "
        "only if the plan is concrete enough."
    )


def build_edit_prompt(run: ContributionRun, design: dict[str, Any], understanding: str) -> str:
    return (
        "OpenSourcePR implementation step 3b: edit the code.\n"
        "Before editing, verify the referenced files and local style one more time.\n"
        "Keep changes within the approved scope unless the repository proves the design inaccurate.\n"
        f"Repository: {run.repo_url}\n"
        f"Selected direction: {run.selected_direction}\n"
        f"Recommended approach: {design.get('recommended')}\n\n"
        f"Understanding checkpoint:\n{understanding}\n\n"
        f"Detailed design:\n{design.get('agent_design') or render_design(design)}"
    )


def build_verification_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    tests = design.get("tests_to_run") or ["run the narrowest relevant pytest command or document why none applies"]
    return (
        "OpenSourcePR implementation step 3c: verify the change.\n"
        "Run focused verification, inspect git diff/status, and report exact commands and results.\n"
        f"Expected tests: {json.dumps(tests, ensure_ascii=False)}\n"
        "Do not open a PR, push, or commit."
    )


def build_implementation_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    return build_edit_prompt(run, design, understanding="Prepared from saved workflow artifacts.")


def implementation_prompt_for_run(*, repo_root: Path, run_id: str) -> str:
    run = load_run(repo_root=repo_root, run_id=run_id)
    return build_implementation_prompt(run, _read_json(run, "02_design.json"))


def validate_implementation_scope(repo_root: Path, design: dict[str, Any]) -> dict[str, Any]:
    changed = [path for path in git_changed_files(repo_root=repo_root) if not path.startswith(".osc_agent/")]
    allowed_files = {str(path).replace("\\", "/") for path in design.get("allowed_files") or []}
    allowed_dirs = [str(path).strip("/\\").replace("\\", "/") for path in design.get("allowed_new_dirs") or []]
    forbidden = design.get("forbidden_paths") or DEFAULT_FORBIDDEN_PATHS

    outside_scope = [
        path for path in changed
        if path not in allowed_files and not any(path == directory or path.startswith(f"{directory}/") for directory in allowed_dirs)
    ]
    forbidden_changes = [path for path in changed if any(Path(path).match(pattern) for pattern in forbidden)]
    added, deleted = git_diff_numstat(repo_root=repo_root)
    max_files = int(design.get("max_changed_files") or 5)
    max_lines = int(design.get("max_diff_lines") or 400)
    violations: list[str] = []
    if not changed:
        violations.append("no implementation files changed")
    if outside_scope:
        violations.append(f"files outside approved scope: {', '.join(outside_scope)}")
    if forbidden_changes:
        violations.append(f"forbidden files changed: {', '.join(forbidden_changes)}")
    if len(changed) > max_files:
        violations.append(f"changed file budget exceeded: {len(changed)} > {max_files}")
    if added + deleted > max_lines:
        violations.append(f"diff line budget exceeded: {added + deleted} > {max_lines}")
    return {
        "ok": not violations,
        "changed_files": changed,
        "added_lines": added,
        "deleted_lines": deleted,
        "outside_scope": outside_scope,
        "forbidden_changes": forbidden_changes,
        "violations": violations,
    }


def record_test_waiver(*, repo_root: Path, run_id: str, reason: str) -> ContributionRun:
    if not reason.strip():
        raise ValueError("test waiver reason is required")
    run = load_run(repo_root=repo_root, run_id=run_id)
    report = _read_json(run, "03_implementation.json")
    if report.get("verification_results"):
        raise ValueError("test waiver is only valid when no verification command is available")
    report["test_waiver"] = {
        "operator": getpass.getuser(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason.strip(),
    }
    scope_ok = bool((report.get("scope_validation") or {}).get("ok"))
    run.final_status = RunStatus.SUCCESS.value if scope_ok else RunStatus.FAILED_VALIDATION.value
    if run.metrics is not None:
        run.metrics["human_confirmations"] = int(run.metrics.get("human_confirmations", 0)) + 1
    if (run.stage_status or {}).get("implement") != StageStatus.RUNNING.value:
        _begin_stage(run, "implement", repo_root)
    _complete_stage(run, "implement", success=scope_ok)
    run.final_status = RunStatus.SUCCESS.value if scope_ok else RunStatus.FAILED_VALIDATION.value
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)
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
    save_run(run)
    _write_metrics_report(run)


def run_verification_commands(
    repo_root: Path,
    commands: list[str],
    *,
    confirm: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """通过统一权限策略执行设计中的验证命令，并为每条命令保存审计日志。"""
    results: list[dict[str, Any]] = []
    log_dir = repo_root / ".osc_agent" / "verification"
    log_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        started = time.perf_counter()
        decision = check_shell_command(command)
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
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            exit_code = completed.returncode
            output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        except subprocess.TimeoutExpired as exc:
            exit_code = -1
            output = f"verification timed out: {exc}"
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_path = log_dir / f"{_content_hash(command)[:12]}.log"
        log_path.write_text(output + "\n", encoding="utf-8")
        results.append(
            {
                "command": command,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "artifact_path": str(log_path),
            }
        )
    return results


def _save_implementation_checkpoint(
    run: ContributionRun,
    step: str,
    output: str,
    *,
    results: list[dict[str, Any]] | None = None,
) -> None:
    checkpoint = run.implementation_checkpoint or {}
    succeeded = results is None or all(item.get("exit_code") == 0 for item in results)
    value: dict[str, Any] = {
        "status": "SUCCEEDED" if succeeded else "FAILED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if output:
        value["output"] = output
    if results is not None:
        value["results"] = results
    checkpoint[step] = value
    run.implementation_checkpoint = checkpoint
    report_path = Path(run.artifacts_dir) / "03_implementation.json"
    if report_path.exists():
        report = _read_json(run, "03_implementation.json")
        report["checkpoint"] = checkpoint
        _write_json(run, "03_implementation.json", report)
    save_run(run)


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
            "final_status": run.final_status,
        }
    )

