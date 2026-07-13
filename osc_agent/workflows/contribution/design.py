from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from osc_agent.harness.permissions import safe_repo_path
from osc_agent.tools.git import git_head
from osc_agent.tools.repo import find_functions
from osc_agent.workflows.contribution.discover import _try_import_stage_agents
from osc_agent.workflows.contribution.models import ContributionRun, DEFAULT_FORBIDDEN_PATHS
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
    discover = _read_json(run, "01_discover.json")
    selected = direction or run.selected_direction or _default_direction(discover)
    _ensure_direction_is_known(selected, discover)
    run.selected_direction = selected

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


def build_design_review_prompt(*, discover: dict[str, Any], selected: str) -> str:
    focused = _focused_discover_for_design(discover, selected)
    return (
        "OpenSourcePR step 2: produce a concrete technical design.\n"
        f"Selected direction: {selected}\n\n"
        "Only use evidence relevant to the selected direction. Ignore other candidate issues unless they "
        "directly explain the selected one.\n\n"
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
    searchable = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for _, path in resolved_files
        if path.is_file()
    )
    missing_symbols = [symbol for symbol in symbols if symbol not in searchable]
    return {
        "ok": not invalid_paths and not missing and not missing_symbols,
        "files": files,
        "invalid_paths": invalid_paths,
        "missing_files": missing,
        "symbols": symbols,
        "missing_symbols": missing_symbols,
    }


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


def _build_design_evidence(
    repo_root: Path,
    allowed_files: list[str],
    target_symbols: list[str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for relative in allowed_files:
        try:
            path = _resolve_design_file(repo_root, relative)
        except ValueError:
            continue
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


def _resolve_design_file(repo_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"design path must be relative to the repository: {relative}")
    return safe_repo_path(repo_root, relative)
