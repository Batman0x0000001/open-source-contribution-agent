from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from osc_agent.harness.contracts import RunStatus
from osc_agent.harness.repository_boundary import safe_repo_path
from osc_agent.tools.git import git_head
from osc_agent.workflows.contribution.agents import run_design_generation
from osc_agent.workflows.contribution.models import (
    AcceptanceCheck,
    ContributionRun,
    ContributionSpec,
    DEFAULT_FORBIDDEN_PATHS,
    DesignContract,
    RequirementKind,
    ScopeContract,
)
from osc_agent.workflows.contribution.state import (
    _content_hash,
    _read_json,
    _require_consistent_run,
    _settings_snapshot,
    _write_json,
    _write_metrics_report,
    _write_text,
    load_run,
    save_run,
)
from osc_agent.workflows.contribution.transitions import _begin_stage, _complete_stage

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
    _begin_stage(run, "design", repo_root)
    try:
        discover = _read_json(run, "01_discover.json")
        review_path = Path(run.artifacts_dir) / "01_discover_human_review.json"
        if review_path.exists():
            discover = {**discover, "human_review": _read_json(run, review_path.name).get("review", "")}
        selected_direction = _resolve_direction(
            direction or run.selected_direction_id or run.selected_direction or _default_direction(discover),
            discover,
        )
        selected = str(selected_direction["name"])
        run.selected_direction = selected
        run.selected_direction_id = str(selected_direction["id"])
        run.selected_issue_number = selected_direction.get("issue_number")

        llm_design = None
        if client is not None and settings is not None:
            llm_design = run_design_generation(
                client,
                settings,
                _focused_discover_for_design(discover, selected),
                selected,
                repo_root=repo_root,
            )

        payload = _design_payload_from_result(
            repo_root=repo_root,
            discover=discover,
            selected=selected,
            selected_direction=selected_direction,
            llm_design=llm_design,
            agent_design=agent_design,
        )
        limits = run.config_snapshot or {}
        payload["max_changed_files"] = int(limits.get("max_changed_files") or payload["max_changed_files"])
        payload["max_diff_lines"] = int(limits.get("max_diff_lines") or payload["max_diff_lines"])
        _validate_design_payload(payload)
        _write_json(run, "02_design.json", payload)
        _write_text(run, "02_design.md", render_design(payload))
        _write_text(run, "02_design_agent_prompt.md", payload["agent_design_prompt"])
        _complete_stage(run, "design", success=True)
    except Exception as exc:
        run.final_status = (
            RunStatus.FAILED_VALIDATION.value if isinstance(exc, ValueError) else RunStatus.FAILED_TOOL.value
        )
        if run.metrics is not None:
            run.metrics["failure_reason"] = str(exc)[:1000]
        if (run.stage_status or {}).get("design") == "RUNNING":
            _complete_stage(run, "design", success=False)
        _write_metrics_report(run)
        raise
    _write_metrics_report(run)
    return run


def attach_design_agent_review(*, repo_root: Path, run_id: str, review: str) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    _require_design_mutable(run)
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
        "task_type",
        "baseline_checks",
        "reproduction_mode",
        "reproduction_test_files",
    }
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise ValueError(f"unsupported design contract fields: {', '.join(unknown)}")
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    _require_design_mutable(run)
    payload = _read_json(run, "02_design.json")
    payload.update(updates)
    if "tests_to_run" in updates and "acceptance_checks" not in updates:
        payload["acceptance_checks"] = []
    if "allowed_files" in updates and "task_type" not in updates:
        payload["task_type"] = ""
    _normalize_scope_contract(payload)
    payload["validation"] = validate_design_files(repo_root, payload)
    discover = _read_json(run, "01_discover.json")
    _refresh_contribution_contract(
        repo_root,
        payload,
        discover,
        str(payload.get("selected_direction") or ""),
        issue_number=payload.get("selected_issue_number"),
    )
    _validate_design_payload(payload)
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
        _require_design_mutable(run)
        payload = _read_json(run, "02_design.json")
        payload["max_changed_files"] = int(run.config_snapshot["max_changed_files"])
        payload["max_diff_lines"] = int(run.config_snapshot["max_diff_lines"])
        _validate_design_payload(payload)
        _write_json(run, "02_design.json", payload)
        _write_text(run, "02_design.md", render_design(payload))
    save_run(run)
    return run


def build_design_review_prompt(*, discover: dict[str, Any], selected: str) -> str:
    focused = _focused_discover_for_design(discover, selected)
    return (
        "OpenSourcePR step 2: produce a concrete technical design.\n"
        f"Selected direction: {selected}\n\n"
        "Only use evidence relevant to the selected direction. Ignore other candidate issues unless they "
        "directly explain the selected one.\n\n"
        "For behavior changes, reuse an existing failing command when available. Otherwise declare a focused "
        "generated regression test path; the workflow will create it before production edits and freeze it after "
        "it proves the failure. Documentation or configuration-only work may use manual checks.\n\n"
        f"Focused discover evidence:\n{json.dumps(focused, ensure_ascii=False, indent=2)[:12000]}"
    )


def validate_design_files(repo_root: Path, design: dict[str, Any]) -> dict[str, Any]:
    files = _extract_design_files(design)
    resolved_files: list[tuple[str, Path]] = []
    invalid_paths: list[str] = []
    for relative in files:
        try:
            resolved_files.append((relative, _resolve_design_file(repo_root, relative)))
        except ValueError:
            invalid_paths.append(relative)
    missing = [relative for relative, path in resolved_files if not path.exists()]
    symbols = [str(item) for item in design.get("target_symbols") or []]
    found_symbols = {
        symbol
        for _, path in resolved_files
        if path.is_file()
        for symbol, _, _ in _target_symbol_ranges(
            path,
            path.read_text(encoding="utf-8", errors="replace"),
            symbols,
        )
    }
    missing_symbols = [symbol for symbol in symbols if symbol not in found_symbols]
    return {
        "ok": not invalid_paths and not missing and not missing_symbols,
        "files": files,
        "invalid_paths": invalid_paths,
        "missing_files": missing,
        "symbols": symbols,
        "missing_symbols": missing_symbols,
    }


def _validate_design_payload(payload: dict[str, Any]) -> None:
    validation = payload.get("validation") or {}
    if not validation.get("ok", False):
        problems = (
            list(validation.get("invalid_paths") or [])
            + list(validation.get("missing_files") or [])
            + list(validation.get("missing_symbols") or [])
        )
        raise ValueError(f"design references invalid repository evidence: {', '.join(problems)}")
    try:
        ScopeContract.model_validate(payload)
        DesignContract.model_validate(payload)
    except ValueError as exc:
        raise ValueError(f"invalid design contract: {exc}") from exc


def _require_design_mutable(run: ContributionRun) -> None:
    if any((run.stage_status or {}).get(stage) != "PENDING" for stage in ("implement", "draft_pr")):
        raise ValueError("design is frozen after implementation starts; explicitly rewind the run first")


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
    spec = payload.get("contribution_spec") or {}
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
        f"**Task type:** {spec.get('task_type', 'not specified')}\n"
        "**Requirements:**\n"
        f"```json\n{json.dumps(spec.get('requirements') or [], ensure_ascii=False, indent=2)}\n```\n"
        "**Pre-change failure baseline:**\n"
        f"```json\n{json.dumps(spec.get('baseline_checks') or [], ensure_ascii=False, indent=2)}\n```\n"
        "**Reproduction plan:**\n"
        f"```json\n{json.dumps(spec.get('reproduction') or {}, ensure_ascii=False, indent=2)}\n```\n"
        f"**Missing file warnings:** {missing or 'none'}\n\n"
        "## Maintainer Comment\n"
        f"{payload['maintainer_comment']}\n\n"
        "## Interview Story\n"
        f"{payload['interview_story']}\n"
    )


def _design_payload_from_result(
    *,
    repo_root: Path,
    discover: dict[str, Any],
    selected: str,
    selected_direction: dict[str, Any],
    llm_design: dict[str, Any] | None,
    agent_design: str | None,
) -> dict[str, Any]:
    template = _template_design(selected, selected_direction, discover, agent_design)
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
            "allowed_files": _provided(
                llm_design,
                "allowed_files",
                llm_design.get("files_to_modify", []),
            ),
            "allowed_new_dirs": _provided(llm_design, "allowed_new_dirs", ["tests"]),
            "forbidden_paths": _provided(llm_design, "forbidden_paths", DEFAULT_FORBIDDEN_PATHS),
            "target_symbols": llm_design.get("target_symbols") or [],
            "requirements": llm_design.get("requirements") or [],
            "acceptance_checks": llm_design.get("acceptance_checks") or [],
            "assumptions": llm_design.get("assumptions") or [],
            "impact_area": llm_design.get("impact_area") or [],
            "max_changed_files": _provided(llm_design, "max_changed_files", 5),
            "max_diff_lines": _provided(llm_design, "max_diff_lines", 400),
            "task_type": llm_design.get("task_type") or "",
            "baseline_checks": llm_design.get("baseline_checks") or [],
            "reproduction_mode": llm_design.get("reproduction_mode") or "",
            "reproduction_test_files": llm_design.get("reproduction_test_files") or [],
        }
    else:
        payload = template
    _normalize_scope_contract(payload)
    payload["validation"] = validate_design_files(repo_root, payload)
    payload["target_symbols"] = list(payload.get("target_symbols") or [])
    payload["assumptions"] = list(payload.get("assumptions") or [])
    payload["impact_area"] = list(payload.get("impact_area") or payload["allowed_files"])
    _refresh_contribution_contract(
        repo_root,
        payload,
        discover,
        selected,
        issue_number=payload.get("selected_issue_number"),
    )
    return payload


def _normalize_scope_contract(payload: dict[str, Any]) -> None:
    scope = ScopeContract.model_validate(payload)
    payload.update(scope.model_dump(mode="json"))


def _provided(payload: dict[str, Any], key: str, default: Any) -> Any:
    return payload[key] if key in payload else default


def _focused_discover_for_design(discover: dict[str, Any], selected: str) -> dict[str, Any]:
    directions = discover.get("top_directions") or []
    selected_direction = next(
        (item for item in directions if item.get("id") == selected or item.get("name") == selected),
        None,
    )
    selected_issue = selected_direction.get("issue_number") if selected_direction else None
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
        "llm_analysis_summary": str(discover.get("llm_analysis_summary") or "")[:2000],
        "human_review": str(discover.get("human_review") or "")[:2000],
    }


def _compact_evidence_pack(evidence_pack: dict[str, Any]) -> dict[str, Any]:
    symbols = evidence_pack.get("symbols") if isinstance(evidence_pack, dict) else {}
    if isinstance(symbols, dict):
        symbols = {name: values[:5] if isinstance(values, list) else values for name, values in symbols.items()}
    return {
        "entrypoints": evidence_pack.get("entrypoints", []),
        "symbols": symbols,
    }


def _template_design(
    selected: str,
    selected_direction: dict[str, Any],
    discover: dict[str, Any],
    agent_design: str | None,
) -> dict[str, Any]:
    return {
        "selected_direction": selected,
        "selected_direction_id": selected_direction["id"],
        "selected_issue_number": selected_direction.get("issue_number"),
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
        "requirements": [],
        "source_evidence": [],
        "acceptance_checks": [],
        "assumptions": ["Target files and symbols must be confirmed before implementation."],
        "impact_area": [],
        "max_changed_files": 5,
        "max_diff_lines": 400,
        "task_type": "",
        "baseline_checks": [],
        "reproduction_mode": "",
        "reproduction_test_files": [],
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


def _resolve_direction(selected: str, discover: dict[str, Any]) -> dict[str, Any]:
    directions = [item for item in discover.get("top_directions", []) if isinstance(item, dict)]
    match = next(
        (item for item in directions if item.get("id") == selected or item.get("name") == selected),
        None,
    )
    if match is None:
        raise ValueError(f"unknown contribution direction: {selected}")
    return match


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


def _build_design_evidence(
    repo_root: Path,
    allowed_files: list[str],
    target_symbols: list[str],
    requirement_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    found_symbols: set[str] = set()
    commit_sha = git_head(repo_root=repo_root).strip()
    for relative in allowed_files:
        try:
            path = _resolve_design_file(repo_root, relative)
        except ValueError as exc:
            raise ValueError(f"invalid design evidence path: {relative}") from exc
        if not path.is_file():
            raise ValueError(
                "allowed_files may contain only existing repository files; planned new pytest files "
                "belong in reproduction_test_files and their parent directory in allowed_new_dirs: "
                f"{relative}"
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        ranges = _target_symbol_ranges(path, text, target_symbols)
        if target_symbols:
            found_symbols.update(symbol for symbol, _, _ in ranges)
        else:
            ranges = [("", 1, max(1, len(lines)))]
        for symbol, start, end in ranges:
            snippet = "\n".join(lines[start - 1 : end])
            evidence.append(
                {
                    "commit_sha": commit_sha,
                    "file": relative.replace("\\", "/"),
                    "symbol": symbol,
                    "line_range": [start, end],
                    "content_hash": _content_hash(snippet),
                    "requirement_ids": list(requirement_ids or []),
                }
            )
    missing_symbols = sorted(set(target_symbols) - found_symbols)
    if missing_symbols:
        raise ValueError(f"target symbols not found: {', '.join(missing_symbols)}")
    return evidence


def _refresh_contribution_contract(
    repo_root: Path,
    payload: dict[str, Any],
    discover: dict[str, Any],
    selected: str,
    *,
    issue_number: int | None = None,
) -> None:
    task_type = _task_type(payload.get("task_type"), selected, list(payload.get("allowed_files") or []))
    requirements = _build_requirements(
        discover,
        selected,
        payload.get("requirements"),
        issue_number=issue_number,
        kind=RequirementKind(task_type),
    )
    requirement_ids = [item["id"] for item in requirements]
    payload["task_type"] = task_type
    payload["baseline_checks"] = _normalize_baseline_checks(payload.get("baseline_checks"))
    reproduction = _normalize_reproduction(
        payload.get("reproduction_mode"),
        payload.get("reproduction_test_files"),
        payload["baseline_checks"],
        list(payload.get("tests_to_run") or []),
    )
    payload["reproduction_mode"] = reproduction["mode"]
    payload["reproduction_test_files"] = reproduction["test_files"]
    payload["source_evidence"] = _build_design_evidence(
        repo_root,
        list(payload.get("allowed_files") or []),
        list(payload.get("target_symbols") or []),
        requirement_ids,
    )
    payload["acceptance_checks"] = _normalize_acceptance_checks(
        payload.get("acceptance_checks"),
        list(payload.get("success_criteria") or []),
        list(payload.get("tests_to_run") or []),
        requirement_ids,
        task_type,
    )
    payload["contribution_spec"] = ContributionSpec.model_validate(
        {
            "task_type": task_type,
            "requirements": requirements,
            "baseline_checks": payload["baseline_checks"],
            "reproduction": reproduction,
        }
    ).model_dump(mode="json")


def _build_requirements(
    discover: dict[str, Any],
    selected: str,
    declared: Any = None,
    *,
    issue_number: int | None = None,
    kind: RequirementKind = RequirementKind.BEHAVIOR,
) -> list[dict[str, str]]:
    source = f"Issue #{issue_number}" if issue_number is not None else "Selected direction"
    explicit = [item for item in declared or [] if isinstance(item, dict)]
    if explicit:
        return [
            {
                "id": f"REQ-{index}",
                "text": str(item.get("text") or "").strip(),
                "source": source,
                "source_excerpt": str(item.get("source_excerpt") or "").strip(),
                "kind": kind.value,
            }
            for index, item in enumerate(explicit, start=1)
        ]
    if issue_number is not None:
        issue = next(
            (item for item in discover.get("candidate_issues") or [] if item.get("number") == issue_number),
            None,
        )
        if issue:
            title = str(issue.get("title") or selected).strip()
            raw_body = str(issue.get("body") or "").strip()
            items = _enumerated_requirements(raw_body)
            if items:
                return [
                    {
                        "id": f"REQ-{index}",
                        "text": item,
                        "source": source,
                        "source_excerpt": item,
                        "kind": kind.value,
                    }
                    for index, item in enumerate(items, start=1)
                ]
            body = " ".join(raw_body.split())[:800]
            text = f"{title}: {body}" if body else title
            return [{
                "id": "REQ-1",
                "text": text,
                "source": source,
                "source_excerpt": body or title,
                "kind": kind.value,
            }]
    direction = next(
        (item for item in discover.get("top_directions") or [] if item.get("name") == selected),
        {},
    )
    description = str(direction.get("description") or selected).strip()
    return [{
        "id": "REQ-1",
        "text": description,
        "source": source,
        "source_excerpt": description,
        "kind": kind.value,
    }]


def _enumerated_requirements(body: str) -> list[str]:
    items: list[str] = []
    for line in body.splitlines():
        match = re.match(r"^\s*(?:[-*+] |\d+[.)]\s+|\[[ xX]\]\s+)(.+?)\s*$", line)
        if match and match.group(1).strip():
            items.append(match.group(1).strip())
    return items


def _task_type(value: Any, selected: str, allowed_files: list[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"behavior", "docs", "config"}:
        return normalized
    text = f"{selected} {' '.join(allowed_files)}".lower()
    if any(marker in text for marker in ("docs", "documentation", "readme", ".md")):
        return "docs"
    config_suffixes = (".toml", ".yaml", ".yml", ".json", ".ini", ".cfg")
    if allowed_files and all(path.lower().endswith(config_suffixes) for path in allowed_files):
        return "config"
    return "behavior"


def _normalize_baseline_checks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    checks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command") or "").strip()
        output_contains = str(item.get("output_contains") or "").strip()
        exit_codes = item.get("expected_exit_codes") or [1]
        if command:
            checks.append(
                {
                    "command": command,
                    "expected_exit_codes": [int(code) for code in exit_codes],
                    "output_contains": output_contains,
                }
            )
    return checks


def _normalize_reproduction(
    mode: Any,
    test_files: Any,
    baseline_checks: list[dict[str, Any]],
    tests_to_run: list[str],
) -> dict[str, Any]:
    normalized_mode = str(mode or "").strip().lower()
    normalized_files = [
        str(path).strip().replace("\\", "/")
        for path in test_files or []
        if str(path).strip()
    ]
    if normalized_mode not in {"existing", "generated_test"}:
        normalized_mode = "existing" if baseline_checks else "generated_test" if normalized_files else ""
    if normalized_mode == "existing":
        command = str((baseline_checks[0] if baseline_checks else {}).get("command") or "")
    else:
        command = str(tests_to_run[0] if tests_to_run else "")
    return {"mode": normalized_mode, "command": command, "test_files": normalized_files}


def _normalize_acceptance_checks(
    value: Any,
    success_criteria: list[str],
    tests_to_run: list[str],
    requirement_ids: list[str],
    task_type: str,
) -> list[dict[str, Any]]:
    checks = [dict(item) for item in value or [] if isinstance(item, dict)]
    if not checks and tests_to_run:
        checks = [
            {"criterion": f"Verification command passes: {command}", "command": command, "manual_check": False}
            for command in tests_to_run
        ]
    if not checks:
        checks = [
            {"criterion": item, "command": "", "manual_check": task_type in {"docs", "config"}}
            for item in success_criteria
        ]
    for check in checks:
        check["criterion"] = str(check.get("criterion") or "").strip()
        check["command"] = str(check.get("command") or "").strip()
        check["manual_check"] = bool(check.get("manual_check", False))
        declared_ids = [str(item) for item in check.get("requirement_ids") or []]
        check["requirement_ids"] = declared_ids or (requirement_ids if len(requirement_ids) == 1 else [])
    expected_ids = set(requirement_ids)
    covered_ids = {
        requirement_id
        for check in checks
        for requirement_id in check["requirement_ids"]
    }
    unknown_ids = sorted(covered_ids - expected_ids)
    missing_ids = sorted(expected_ids - covered_ids)
    if unknown_ids or missing_ids:
        details: list[str] = []
        if unknown_ids:
            details.append(f"unknown: {', '.join(unknown_ids)}")
        if missing_ids:
            details.append(f"missing: {', '.join(missing_ids)}")
        raise ValueError(f"acceptance_checks requirement coverage is invalid ({'; '.join(details)})")
    normalized: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        try:
            normalized.append(AcceptanceCheck.model_validate(check).model_dump(mode="json"))
        except ValueError as exc:
            raise ValueError(f"acceptance_checks[{index}] is invalid: {exc}") from exc
    return normalized


def _target_symbol_ranges(path: Path, text: str, target_symbols: list[str]) -> list[tuple[str, int, int]]:
    if not target_symbols:
        return []
    wanted = set(target_symbols)
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        return _PythonSymbolVisitor(wanted).collect(tree)
    ranges: list[tuple[str, int, int]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for symbol in wanted:
            if symbol in line:
                ranges.append((symbol, number, number))
    return ranges


class _PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self, wanted: set[str]) -> None:
        self.wanted = wanted
        self.class_path: list[str] = []
        self.function_depth = 0
        self.ranges: list[tuple[str, int, int]] = []

    def collect(self, tree: ast.AST) -> list[tuple[str, int, int]]:
        self.visit(tree)
        return self.ranges

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, node.name, ".".join([*self.class_path, node.name]))
        self.class_path.append(node.name)
        self.generic_visit(node)
        self.class_path.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = (
            ".".join([*self.class_path, node.name])
            if self.class_path and self.function_depth == 0
            else node.name
        )
        self._record(node, node.name, qualified)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def _record(self, node: ast.AST, *candidates: str) -> None:
        for symbol in dict.fromkeys(candidates):
            if symbol in self.wanted:
                line = int(getattr(node, "lineno", 1))
                self.ranges.append((symbol, line, int(getattr(node, "end_lineno", line))))


def _resolve_design_file(repo_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"design path must be relative to the repository: {relative}")
    return safe_repo_path(repo_root, relative)
