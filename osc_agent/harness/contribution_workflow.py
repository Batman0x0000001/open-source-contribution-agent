from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from osc_agent.harness.tasks import create_default_task_graph
from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.harness.todo import todo_write
from osc_agent.tools.github import (
    CANDIDATE_LABELS,
    apply_issue_scores,
    fetch_issue_comments,
    fetch_issue_activity,
    fetch_issues,
    filter_candidate_issues,
    load_issues_file,
)
from osc_agent.tools.git import git_changed_files, git_diff_numstat, git_head, git_status
from osc_agent.tools.pr import draft_pr
from osc_agent.tools.repo import (
    analyze_architecture_dimensions,
    collect_repo_evidence_pack,
    detect_entrypoints,
    find_functions,
    inspect_repo,
    repo_tree,
)

STAGES = {"discover", "design", "implement", "draft_pr"}
STATE_SCHEMA_VERSION = 2
DEFAULT_FORBIDDEN_PATHS = [
    ".git/**",
    ".github/**",
    ".env*",
    "*lock*",
    "**/security/**",
]


@dataclass
class ContributionRun:
    run_id: str
    repo_root: str
    repo_url: str
    stage: str
    selected_direction: str | None
    artifacts_dir: str
    schema_version: int = STATE_SCHEMA_VERSION
    base_commit_sha: str = ""
    issue_snapshot_at: str = ""
    config_snapshot: dict[str, Any] | None = None
    stage_status: dict[str, str] | None = None
    stage_hashes: dict[str, str] | None = None
    critical_file_hashes: dict[str, str] | None = None
    final_status: str | None = None
    metrics: dict[str, Any] | None = None


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
    )
    save_run(run)
    return run


def load_run(*, repo_root: Path, run_id: str) -> ContributionRun:
    path = _runs_dir(repo_root) / run_id / "run.json"
    if not path.exists():
        raise ValueError(f"contribution run not found: {run_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported contribution run schema; expected {STATE_SCHEMA_VERSION}. Start a new run."
        )
    return ContributionRun(**data)


def save_run(run: ContributionRun) -> None:
    artifacts = Path(run.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    _write_raw_json(artifacts / "run.json", asdict(run))


def discover_stage(
    *,
    repo_root: Path,
    repo_url: str,
    issues_file: Path | None = None,
    client: Any | None = None,
    settings: Any | None = None,
    agent_review: str | None = None,
) -> ContributionRun:
    run = create_run(repo_root=repo_root, repo_url=repo_url, settings=settings)
    _begin_stage(run, "discover")
    issues, comments_by_issue, issue_error = _collect_issues(repo_url=repo_url, issues_file=issues_file)
    candidates = filter_candidate_issues(issues, comments_by_issue)
    dimensions = analyze_architecture_dimensions(repo_root=repo_root)
    evidence_pack = build_discover_evidence(repo_root=repo_root)
    issue_scores: list[dict[str, Any]] = []

    llm_result = None
    stage_agents = _try_import_stage_agents()
    if client is not None and settings is not None and stage_agents is not None:
        issue_scores = stage_agents.score_candidate_issues(
            client,
            settings,
            candidates,
            comments_by_issue,
            repo_root=repo_root,
        )
        candidates = apply_issue_scores(candidates, issue_scores)
        llm_result = stage_agents.run_discover_analysis(
            client,
            settings,
            {
                "repo_url": repo_url,
                "repo_overview": inspect_repo(repo_root=repo_root),
                "tree": repo_tree(repo_root=repo_root, depth=3),
                "entrypoints": detect_entrypoints(repo_root=repo_root),
                "candidate_issues": candidates,
                "issue_scores": issue_scores,
                "architecture_dimensions": dimensions,
                "evidence_pack": evidence_pack,
            },
            repo_root=repo_root,
        )

    if llm_result:
        directions = _normalize_directions(llm_result.get("top_directions")) or _top_directions(candidates, dimensions)
        architecture_dimensions = llm_result.get("architecture_insights") or dimensions
        analysis_summary = llm_result.get("analysis_summary") or ""
    else:
        directions = _top_directions(candidates, dimensions)
        architecture_dimensions = dimensions
        analysis_summary = agent_review or ""

    payload = {
        "repo_url": repo_url,
        "repo_overview": inspect_repo(repo_root=repo_root),
        "tree": repo_tree(repo_root=repo_root, depth=3),
        "entrypoints": detect_entrypoints(repo_root=repo_root),
        "candidate_issues": candidates,
        "issue_scores": issue_scores,
        "architecture_dimensions": architecture_dimensions,
        "top_directions": directions,
        "issue_error": issue_error,
        "evidence_pack": evidence_pack,
        "repository_profile": evidence_pack.get("repository_profile", {}),
        "agent_review": analysis_summary,
        "agent_review_prompt": build_discover_review_prompt(
            repo_url=repo_url,
            candidates=candidates,
            dimensions=dimensions,
            evidence_pack=evidence_pack,
        ),
    }
    _write_json(run, "01_discover.json", payload)
    _write_text(run, "01_discover.md", render_discover(payload))
    _write_text(run, "01_discover_agent_prompt.md", payload["agent_review_prompt"])
    run.critical_file_hashes = _evidence_file_hashes(repo_root, evidence_pack)
    _complete_stage(run, "discover", success=True)
    save_run(run)
    _write_metrics_report(run)
    return run


def attach_discover_agent_review(*, repo_root: Path, run_id: str, review: str) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    payload = _read_json(run, "01_discover.json")
    payload["agent_review"] = review
    _write_json(run, "01_discover.json", payload)
    _write_text(run, "01_discover.md", render_discover(payload))
    save_run(run)
    return run


def design_stage(
    *,
    repo_root: Path,
    run_id: str,
    direction: str | None = None,
    client: Any | None = None,
    settings: Any | None = None,
    agent_design: str | None = None,
) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    _begin_stage(run, "design")
    discover = _read_json(run, "01_discover.json")
    selected = direction or run.selected_direction or _default_direction(discover)
    _ensure_direction_is_known(selected, discover)
    run.selected_direction = selected
    run.stage = "design"

    stage_agents = _try_import_stage_agents()
    llm_design = None
    if client is not None and settings is not None and stage_agents is not None:
        llm_design = stage_agents.run_design_generation(client, settings, discover, selected, repo_root=repo_root)

    payload = _design_payload_from_result(
        repo_root=repo_root,
        discover=discover,
        selected=selected,
        llm_design=llm_design,
        agent_design=agent_design,
    )
    limits = run.config_snapshot or {}
    payload["max_changed_files"] = int(limits.get("max_changed_files") or payload["max_changed_files"])
    payload["max_diff_lines"] = int(limits.get("max_diff_lines") or payload["max_diff_lines"])
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    _write_text(run, "02_design_agent_prompt.md", payload["agent_design_prompt"])
    _complete_stage(run, "design", success=True)
    save_run(run)
    _write_metrics_report(run)
    return run


def attach_design_agent_review(*, repo_root: Path, run_id: str, review: str) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    payload = _read_json(run, "02_design.json")
    payload["agent_design"] = review
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    save_run(run)
    return run


def update_design_contract(*, repo_root: Path, run_id: str, updates: dict[str, Any]) -> ContributionRun:
    allowed = {
        "files_to_modify",
        "tests_to_run",
        "allowed_files",
        "allowed_new_dirs",
        "forbidden_paths",
        "target_symbols",
        "acceptance_checks",
        "assumptions",
        "impact_area",
        "max_changed_files",
        "max_diff_lines",
    }
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise ValueError(f"unsupported design contract fields: {', '.join(unknown)}")
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    payload = _read_json(run, "02_design.json")
    payload.update(updates)
    payload["validation"] = validate_design_files(repo_root, payload)
    payload["source_evidence"] = _build_design_evidence(
        repo_root,
        list(payload.get("allowed_files") or []),
        list(payload.get("target_symbols") or []),
    )
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    save_run(run)
    return run


def configure_run(*, repo_root: Path, run_id: str, settings: Any) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    run.config_snapshot = _settings_snapshot(settings)
    design_path = Path(run.artifacts_dir) / "02_design.json"
    if design_path.exists():
        payload = _read_json(run, "02_design.json")
        payload["max_changed_files"] = int(run.config_snapshot["max_changed_files"])
        payload["max_diff_lines"] = int(run.config_snapshot["max_diff_lines"])
        _write_json(run, "02_design.json", payload)
        _write_text(run, "02_design.md", render_design(payload))
    save_run(run)
    return run


def prepare_implementation_stage(*, repo_root: Path, run_id: str) -> tuple[ContributionRun, str]:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    _begin_stage(run, "implement")
    design = _read_json(run, "02_design.json")
    run.stage = "implement"
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
    }
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
) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    existing = _read_text(run, "03_implementation_report.md", default="")
    report = {
        "selected_direction": run.selected_direction,
        "recommended": design.get("recommended"),
        "implementation_prompt": build_implementation_prompt(run, design),
        "created_tasks": [],
        "understanding_output": understanding_output or "",
        "agent_output": agent_output or "Implementation finished without captured output.",
        "verification_output": verification_output or "",
        "git_status_before": _extract_code_block(existing) or "",
        "git_status_after": git_status(repo_root=repo_root),
        "test_summary": test_summary or _infer_test_summary("\n".join([agent_output or "", verification_output or ""])),
    }
    report["verification_results"] = _run_verification_commands(repo_root, design.get("tests_to_run") or [])
    report["scope_validation"] = validate_implementation_scope(repo_root, design)
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
    """Run the implementation substeps in order with an explicit edit checkpoint."""
    run, fallback_prompt = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")

    try:
        understanding = run_step("understanding", build_understanding_prompt(run, design))
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
    try:
        agent_output = run_step("edit", edit_prompt)
        verification = run_step("verification", build_verification_prompt(run, design))
    except Exception as exc:
        _fail_implementation_run(run, exc)
        raise
    return record_implementation_result(
        repo_root=repo_root,
        run_id=run_id,
        understanding_output=understanding,
        agent_output=agent_output,
        verification_output=verification,
    )


def implement_stage(*, repo_root: Path, run_id: str, agent_output: str | None = None) -> ContributionRun:
    if agent_output is None:
        run, _ = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
        return run
    return record_implementation_result(repo_root=repo_root, run_id=run_id, agent_output=agent_output)


def draft_pr_stage(
    *,
    repo_root: Path,
    run_id: str,
    client: Any | None = None,
    settings: Any | None = None,
) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root, check_evidence=False)
    if run.final_status != RunStatus.SUCCESS.value:
        raise ValueError(f"PR draft blocked by implementation status: {run.final_status or 'not validated'}")
    _begin_stage(run, "draft_pr")
    run.stage = "draft_pr"
    _write_text(run, "04_pr_draft.md", draft_pr(repo_root=repo_root, run_id=run_id, client=client, settings=settings))
    _complete_stage(run, "draft_pr", success=True)
    save_run(run)
    _write_metrics_report(run)
    return run


def build_discover_evidence(*, repo_root: Path) -> dict[str, Any]:
    return collect_repo_evidence_pack(repo_root=repo_root)


def build_discover_review_prompt(
    *,
    repo_url: str,
    candidates: list[dict[str, Any]],
    dimensions: list[dict[str, str]],
    evidence_pack: dict[str, Any],
) -> str:
    return (
        "OpenSourcePR step 1: find contribution entry points.\n"
        f"Repository: {repo_url}\n\n"
        f"Candidate issues:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        f"Architecture dimensions:\n{json.dumps(dimensions, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence pack:\n{json.dumps(evidence_pack, ensure_ascii=False, indent=2)}\n"
    )


def build_design_review_prompt(*, discover: dict[str, Any], selected: str) -> str:
    focused = _focused_discover_for_design(discover, selected)
    return (
        "OpenSourcePR step 2: produce a concrete technical design.\n"
        f"Selected direction: {selected}\n\n"
        "Only use evidence relevant to the selected direction. Ignore other candidate issues unless they "
        "directly explain the selected one.\n\n"
        f"Focused discover evidence:\n{json.dumps(focused, ensure_ascii=False, indent=2)[:12000]}"
    )


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


def validate_design_files(repo_root: Path, design: dict[str, Any]) -> dict[str, Any]:
    files = _extract_design_files(design)
    missing = [path for path in files if not (repo_root / path).exists()]
    symbols = [str(item) for item in design.get("target_symbols") or []]
    searchable = "\n".join(
        (repo_root / path).read_text(encoding="utf-8", errors="replace")
        for path in files
        if (repo_root / path).is_file()
    )
    missing_symbols = [symbol for symbol in symbols if symbol not in searchable]
    return {
        "ok": not missing and not missing_symbols,
        "files": files,
        "missing_files": missing,
        "symbols": symbols,
        "missing_symbols": missing_symbols,
    }


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
    if run.stage_status is not None:
        run.stage_status["implement"] = StageStatus.SUCCEEDED.value if scope_ok else StageStatus.FAILED.value
    _write_json(run, "03_implementation.json", report)
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)
    _write_metrics_report(run)
    return run


def render_discover(payload: dict[str, Any]) -> str:
    issue_rows = "\n".join(
        f"| #{issue.get('number')} | {issue.get('title', '')} | {', '.join(issue.get('labels', []))} |"
        for issue in payload["candidate_issues"]
    ) or "| - | No matching issue found | - |"
    directions = "\n\n".join(
        "\n".join(
            [
                f"**Rank {index}: {item.get('name', 'Untitled direction')}**",
                f"- Description: {item.get('description', '')}",
                f"- Source: {item.get('source', '')}",
                f"- Entry: {item.get('entry', '')}",
                f"- Effort: {item.get('effort', '')}",
                f"- Interview value: {item.get('interview', '')}",
                f"- Risk: {item.get('risk', 'Needs maintainer confirmation')}",
            ]
        )
        for index, item in enumerate(payload["top_directions"], start=1)
    )
    dimensions = "\n\n".join(
        "\n".join(
            [
                f"### {item.get('dimension', 'Dimension')}",
                f"- Current: {item.get('current', '')}",
                f"- Gap: {item.get('gap', '')}",
                f"- Impact: {item.get('impact', '')}",
                f"- Improvement: {item.get('improvement', '')}",
                f"- Location: {item.get('location', '')}",
            ]
        )
        for item in payload["architecture_dimensions"]
    )
    review = f"\n## Agent Analysis\n\n{payload['agent_review']}\n" if payload.get("agent_review") else ""
    return (
        "# Open Source Contribution Analysis\n\n"
        f"## Project\nRepository: {payload['repo_url']}\n\n"
        f"## Preparation\n```text\n{payload['repo_overview']}\n\n{payload['tree']}\n```\n\n"
        f"Entrypoints: {', '.join(payload['entrypoints']) or 'not found'}\n\n"
        "## Issue Candidates\n"
        "| Issue | Title | Labels |\n|---|---|---|\n"
        f"{issue_rows}\n\n"
        "## Architecture Gap Analysis\n"
        f"{dimensions}\n"
        f"{review}\n"
        "## Top 3 Contribution Suggestions\n\n"
        f"{directions}\n"
    )


def render_design(payload: dict[str, Any]) -> str:
    options = "\n\n".join(
        f"### {option.get('name', 'Option')}\n"
        f"**Idea:** {option.get('idea', '')}\n"
        f"**Pros:** {option.get('pros', '')}\n"
        f"**Cons:** {option.get('cons', '')}"
        for option in payload["options"]
    )
    validation = payload.get("validation", {})
    missing = ", ".join(validation.get("missing_files", [])) if isinstance(validation, dict) else ""
    return (
        "# 技术方案设计\n\n"
        "## Problem Boundary\n"
        f"**Core problem:** {payload['problem_boundary']}\n"
        f"**Out of scope:** {'; '.join(payload['out_of_scope'])}\n"
        f"**Success criteria:** {'; '.join(payload['success_criteria'])}\n\n"
        "## Design Options\n"
        f"{options}\n\n"
        f"**Recommended:** {payload['recommended']}\n\n"
        "## Implementation Plan\n"
        f"{payload.get('agent_design') or 'Use the recommended scoped implementation.'}\n\n"
        f"**Files to modify:** {', '.join(payload.get('files_to_modify') or ['not specified'])}\n"
        f"**Tests to run:** {', '.join(payload.get('tests_to_run') or ['not specified'])}\n"
        f"**Missing file warnings:** {missing or 'none'}\n\n"
        "## Maintainer Comment\n"
        f"{payload['maintainer_comment']}\n\n"
        "## Interview Story\n"
        f"{payload['interview_story']}\n"
    )


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


def _collect_issues(
    *,
    repo_url: str,
    issues_file: Path | None,
) -> tuple[list[dict[str, Any]], dict[int | str, list[dict[str, Any]]], str | None]:
    if issues_file is not None:
        issues, comments = load_issues_file(str(issues_file))
        return issues, comments, None
    issue_result = fetch_issues(repo_url, labels=sorted(CANDIDATE_LABELS))
    if not issue_result["ok"]:
        return [], {}, issue_result["error"]
    comments: dict[int | str, list[dict[str, Any]]] = {}
    for issue in issue_result["issues"][:20]:
        number = issue.get("number")
        if number is None:
            continue
        result = fetch_issue_comments(repo_url, int(number))
        comments[number] = result.get("comments", [])
        issue["activity"] = fetch_issue_activity(repo_url, int(number))
    return issue_result["issues"], comments, None


def _top_directions(candidates: list[dict[str, Any]], dimensions: list[dict[str, str]]) -> list[dict[str, str]]:
    directions: list[dict[str, str]] = []
    for issue in candidates[:3]:
        directions.append(
            {
                "name": f"Issue #{issue['number']}: {issue['title']}",
                "description": "Small scoped fix or enhancement from an existing issue.",
                "source": f"Issue #{issue['number']}",
                "entry": issue.get("url") or "issue",
                "effort": "small",
                "interview": "Shows requirement clarification, scope control, and verification.",
                "risk": "Needs maintainer confirmation.",
            }
        )
    for item in dimensions:
        if len(directions) >= 3:
            break
        directions.append(
            {
                "name": f"Improve {item['dimension']}",
                "description": item["improvement"],
                "source": f"Code analysis - {item['dimension']}",
                "entry": item["location"],
                "effort": "medium",
                "interview": item["interview_angle"],
                "risk": "Scope must be validated against maintainer expectations.",
            }
        )
    return directions[:3]


def _design_payload_from_result(
    *,
    repo_root: Path,
    discover: dict[str, Any],
    selected: str,
    llm_design: dict[str, Any] | None,
    agent_design: str | None,
) -> dict[str, Any]:
    template = _template_design(selected, discover, agent_design)
    if llm_design:
        payload = {
            **template,
            "problem_boundary": llm_design.get("problem_boundary") or template["problem_boundary"],
            "out_of_scope": llm_design.get("out_of_scope") or template["out_of_scope"],
            "success_criteria": llm_design.get("success_criteria") or template["success_criteria"],
            "options": llm_design.get("options") or template["options"],
            "recommended": llm_design.get("recommended") or template["recommended"],
            "maintainer_comment": llm_design.get("maintainer_comment") or template["maintainer_comment"],
            "interview_story": llm_design.get("interview_story") or template["interview_story"],
            "agent_design": llm_design.get("implementation_plan") or template["agent_design"],
            "files_to_modify": llm_design.get("files_to_modify") or _extract_design_files(llm_design),
            "tests_to_run": llm_design.get("tests_to_run") or _extract_tests_to_run(llm_design),
            "allowed_files": llm_design.get("allowed_files") or llm_design.get("files_to_modify") or [],
            "allowed_new_dirs": llm_design.get("allowed_new_dirs") or ["tests"],
            "forbidden_paths": llm_design.get("forbidden_paths") or DEFAULT_FORBIDDEN_PATHS,
            "target_symbols": llm_design.get("target_symbols") or [],
            "acceptance_checks": llm_design.get("acceptance_checks") or [],
            "assumptions": llm_design.get("assumptions") or [],
            "impact_area": llm_design.get("impact_area") or [],
            "max_changed_files": llm_design.get("max_changed_files") or 5,
            "max_diff_lines": llm_design.get("max_diff_lines") or 400,
        }
    else:
        payload = template
    payload["validation"] = validate_design_files(repo_root, payload)
    payload["allowed_files"] = list(payload.get("allowed_files") or payload.get("files_to_modify") or [])
    payload["allowed_new_dirs"] = list(payload.get("allowed_new_dirs") or ["tests"])
    payload["forbidden_paths"] = list(payload.get("forbidden_paths") or DEFAULT_FORBIDDEN_PATHS)
    payload["target_symbols"] = list(payload.get("target_symbols") or [])
    payload["source_evidence"] = _build_design_evidence(
        repo_root,
        payload["allowed_files"],
        payload["target_symbols"],
    )
    payload["acceptance_checks"] = list(
        payload.get("acceptance_checks")
        or [{"criterion": item, "command": "", "manual_check": True} for item in payload.get("success_criteria", [])]
    )
    payload["assumptions"] = list(payload.get("assumptions") or [])
    payload["impact_area"] = list(payload.get("impact_area") or payload["allowed_files"])
    payload["max_changed_files"] = int(payload.get("max_changed_files") or 5)
    payload["max_diff_lines"] = int(payload.get("max_diff_lines") or 400)
    return payload


def _focused_discover_for_design(discover: dict[str, Any], selected: str) -> dict[str, Any]:
    directions = discover.get("top_directions") or []
    selected_direction = next((item for item in directions if item.get("name") == selected), None)
    selected_issue = _issue_from_selected_direction(selected)
    candidate_issues = discover.get("candidate_issues") or []
    issue_scores = discover.get("issue_scores") or []
    if selected_issue is not None:
        candidate_issues = [issue for issue in candidate_issues if issue.get("number") == selected_issue]
        issue_scores = [score for score in issue_scores if score.get("number") == selected_issue]
    else:
        candidate_issues = [
            issue for issue in candidate_issues
            if str(issue.get("title", "")).lower() in selected.lower()
            or selected.lower() in str(issue.get("title", "")).lower()
        ][:1]
    return {
        "repo_url": discover.get("repo_url"),
        "selected_direction": selected_direction or {"name": selected},
        "candidate_issues": candidate_issues,
        "issue_scores": issue_scores,
        "entrypoints": discover.get("entrypoints", []),
        "architecture_dimensions": discover.get("architecture_dimensions", [])[:7],
        "evidence_pack": _compact_evidence_pack(discover.get("evidence_pack") or {}),
        "agent_review": str(discover.get("agent_review") or "")[:2000],
    }


def _issue_from_selected_direction(selected: str) -> int | None:
    match = re.search(r"Issue\s*#(\d+)", selected, flags=re.I)
    return int(match.group(1)) if match else None


def _compact_evidence_pack(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    symbols = evidence_pack.get("symbols") if isinstance(evidence_pack, dict) else {}
    if isinstance(symbols, dict):
        symbols = {name: values[:5] if isinstance(values, list) else values for name, values in symbols.items()}
    return {
        "entrypoints": evidence_pack.get("entrypoints", []),
        "symbols": symbols,
    }


def _template_design(selected: str, discover: dict[str, Any], agent_design: str | None) -> dict[str, Any]:
    return {
        "selected_direction": selected,
        "problem_boundary": f'Complete a small, reviewable open source contribution around "{selected}".',
        "out_of_scope": [
            "Do not push, commit, or open a PR automatically.",
            "Do not introduce GitHub write operations.",
            "Do not perform broad cross-module refactors.",
        ],
        "success_criteria": [
            "Changes stay near 1-3 core files.",
            "Focused tests or clear manual verification are provided.",
            "The PR draft explains Problem, Solution, Testing, and reviewer notes.",
        ],
        "options": _design_options(selected, discover),
        "recommended": "Option 1: smallest reviewable extension",
        "maintainer_comment": _maintainer_comment(selected),
        "interview_story": _interview_story(selected),
        "agent_design": agent_design or "",
        "files_to_modify": [],
        "tests_to_run": [],
        "allowed_files": [],
        "allowed_new_dirs": ["tests"],
        "forbidden_paths": DEFAULT_FORBIDDEN_PATHS,
        "target_symbols": [],
        "source_evidence": [],
        "acceptance_checks": [],
        "assumptions": ["Target files and symbols must be confirmed before implementation."],
        "impact_area": [],
        "max_changed_files": 5,
        "max_diff_lines": 400,
        "agent_design_prompt": build_design_review_prompt(discover=discover, selected=selected),
    }


def _design_options(selected: str, discover: dict[str, Any]) -> list[dict[str, str]]:
    entry = next((item["entry"] for item in discover.get("top_directions", []) if item["name"] == selected), "TBD")
    return [
        {
            "name": "Option 1: smallest reviewable extension",
            "idea": f"Start near {entry}, reuse existing abstractions, and add only necessary code and tests.",
            "pros": "Small diff and easy review.",
            "cons": "Limited coverage; follow-up PRs may be needed.",
        },
        {
            "name": "Option 2: strategy extraction",
            "idea": "Extract related behavior into a helper or strategy for future extension.",
            "pros": "More extensible.",
            "cons": "Higher implementation and review complexity.",
        },
        {
            "name": "Option 3: tests and docs first",
            "idea": "Add focused tests or docs before broader functionality.",
            "pros": "Low risk and maintainer-friendly.",
            "cons": "Less feature depth.",
        },
    ]


def _normalize_directions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and item.get("name") and item.get("description")]


def _extract_design_files(design: dict[str, Any]) -> list[str]:
    explicit = design.get("files_to_modify")
    if isinstance(explicit, list):
        return sorted({str(item).strip() for item in explicit if str(item).strip()})
    text = json.dumps(design, ensure_ascii=False, default=str)
    matches = re.findall(r"(?<![\w./-])([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|md|toml|json|yaml|yml))(?![\w./-])", text)
    return sorted(set(matches))


def _extract_tests_to_run(design: dict[str, Any]) -> list[str]:
    explicit = design.get("tests_to_run")
    if isinstance(explicit, list):
        return [str(item) for item in explicit if str(item).strip()]
    text = str(design.get("implementation_plan") or "")
    commands = re.findall(r"(?:python -m pytest|pytest)[^\n`]*", text)
    return [command.strip() for command in commands]


def _default_direction(discover: dict[str, Any]) -> str:
    directions = discover.get("top_directions") or []
    if not directions:
        raise ValueError("discover artifact has no contribution directions")
    return directions[0]["name"]


def _ensure_direction_is_known(selected: str, discover: dict[str, Any]) -> None:
    known = {item["name"] for item in discover.get("top_directions", [])}
    if known and selected not in known:
        return


def _maintainer_comment(selected: str) -> str:
    return (
        f"I noticed a scoped opportunity around {selected}. I am considering a small PR that preserves the "
        "current public API, adds focused tests, and documents the behavior. Would this direction be useful "
        "for the project, or is there an existing plan I should align with?"
    )


def _interview_story(selected: str) -> str:
    return (
        f"I first used issue context and architecture evidence to identify {selected}, then compared a minimal "
        "extension, a strategy extraction, and a tests/docs-first approach. I chose the smallest reviewable path "
        "to demonstrate scope control, API design, and verification-driven implementation."
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


def _try_import_stage_agents():
    try:
        from osc_agent.harness import stage_agents

        return stage_agents
    except ImportError:
        return None


def _runs_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "contribution_runs"


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


def _build_design_evidence(
    repo_root: Path,
    allowed_files: list[str],
    target_symbols: list[str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for relative in allowed_files:
        path = repo_root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        evidence.append(
            {
                "commit_sha": git_head(repo_root=repo_root).strip(),
                "file": relative.replace("\\", "/"),
                "symbol": next((name for name in target_symbols if name in text), ""),
                "line_range": [1, max(1, len(lines))],
                "content_hash": _content_hash(text),
            }
        )
    return evidence


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


def _begin_stage(run: ContributionRun, stage: str) -> None:
    if run.stage_status is None or run.metrics is None:
        raise ValueError("run state is missing required schema fields")
    run.stage = stage
    run.stage_status[stage] = StageStatus.RUNNING.value
    stages = run.metrics.setdefault("stages", {})
    stages[stage] = {"started_at": datetime.now(timezone.utc).isoformat()}
    save_run(run)


def _complete_stage(run: ContributionRun, stage: str, *, success: bool) -> None:
    if run.stage_status is None or run.metrics is None:
        return
    run.stage_status[stage] = StageStatus.SUCCEEDED.value if success else StageStatus.FAILED.value
    record = run.metrics.setdefault("stages", {}).setdefault(stage, {})
    finished = datetime.now(timezone.utc)
    record["finished_at"] = finished.isoformat()
    try:
        started = datetime.fromisoformat(str(record["started_at"]))
        record["duration_ms"] = int((finished - started).total_seconds() * 1000)
    except (KeyError, ValueError):
        record["duration_ms"] = 0


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


def _run_verification_commands(repo_root: Path, commands: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    log_dir = repo_root / ".osc_agent" / "verification"
    log_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        started = time.perf_counter()
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
