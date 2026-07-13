from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from osc_agent.tools.git import git_diff, git_status
from osc_agent.workflows.contribution.models import ContributionRun
from osc_agent.workflows.contribution.state import (
    _read_json,
    _read_text,
    _require_consistent_run,
    _write_metrics_report,
    _write_text,
    load_run,
    save_run,
)
from osc_agent.workflows.contribution.transitions import _begin_stage, _complete_stage

def draft_pr_stage(
    *,
    repo_root: Path,
    run_id: str,
    client: Any | None = None,
    settings: Any | None = None,
) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root, check_evidence=False)
    _begin_stage(run, "draft_pr", repo_root)
    _write_text(
        run,
        "04_pr_draft.md",
        build_workflow_pr_draft(repo_root=repo_root, run_id=run_id, client=client, settings=settings),
    )
    _complete_stage(run, "draft_pr", success=True)
    save_run(run)
    _write_metrics_report(run)
    return run


def build_workflow_pr_draft(
    *,
    repo_root: Path,
    run_id: str,
    client: Any | None = None,
    settings: Any | None = None,
) -> str:
    """基于权威 Run 产物和当前实现 worktree 生成 PR 草稿。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    discover = _read_json(run, "01_discover.json")
    design = _read_json(run, "02_design.json")
    implementation = _read_text(run, "03_implementation_report.md")
    diff = git_diff(repo_root=repo_root)
    status = git_status(repo_root=repo_root)
    changed_files = _changed_files(diff, status)
    selected = str(design.get("selected_direction") or _first_direction_name(discover))
    llm_draft = _try_llm_pr_draft(
        repo_root=repo_root,
        client=client,
        settings=settings,
        selected_direction=selected,
        design=design,
        implementation=implementation,
        diff=diff,
        changed_files=changed_files,
    )
    if llm_draft:
        return llm_draft
    changes = "\n".join(f"- Updated `{path}`" for path in changed_files) or "- No local file changes detected yet."
    testing = _extract_section(implementation, "Testing") or "No explicit test result captured. Run focused tests before submitting."
    solution = design.get("agent_design") or design.get("recommended") or "Use the recommended scoped implementation plan."
    notes = _reviewer_notes(design, implementation)
    return (
        "标题：\n"
        f"`{_workflow_title(selected, changed_files)}`\n\n"
        "**Problem**\n"
        f"{design.get('problem_boundary', selected)}\n\n"
        "**Solution**\n"
        f"{solution}\n\n"
        "**Changes**\n"
        f"{changes}\n\n"
        "**Testing**\n"
        f"{testing}\n\n"
        "**Notes for Reviewer**\n"
        f"{notes}"
    )


def _try_llm_pr_draft(
    *,
    repo_root: Path,
    client: Any | None,
    settings: Any | None,
    selected_direction: str,
    design: dict,
    implementation: str,
    diff: str,
    changed_files: list[str],
) -> str | None:
    if client is None or settings is None:
        return None
    from osc_agent.workflows.contribution.agents import run_pr_draft_generation

    result = run_pr_draft_generation(
        client,
        settings,
        {
            "selected_direction": selected_direction,
            "design_summary": json.dumps(design, ensure_ascii=False, indent=2),
            "implementation_report": implementation,
            "git_diff": diff,
            "changed_files": changed_files,
        },
        repo_root=repo_root,
    )
    if not result:
        return None
    changes = "\n".join(f"- {item}" for item in result.get("changes", [])) or "- No local file changes detected yet."
    notes = "\n".join(f"- {item}" for item in result.get("reviewer_notes", [])) or "- Review the saved workflow artifacts."
    return (
        "Title:\n"
        f"`{result.get('title', _workflow_title(selected_direction, changed_files))}`\n\n"
        "**Problem**\n"
        f"{result.get('problem', selected_direction)}\n\n"
        "**Solution**\n"
        f"{result.get('solution', 'See the saved implementation report.')}\n\n"
        "**Changes**\n"
        f"{changes}\n\n"
        "**Testing**\n"
        f"{result.get('testing', 'No explicit test result captured.')}\n\n"
        "**Notes for Reviewer**\n"
        f"{notes}"
    )


def _changed_files(diff: str, status: str) -> list[str]:
    files: set[str] = set()
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", diff, flags=re.MULTILINE):
        files.add(match.group(2))
    for line in status.splitlines():
        if not line.strip() or line == "(no output)":
            continue
        path = line[3:].strip() if len(line) > 3 else line.strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.add(path)
    return sorted(files)


def _first_direction_name(discover: dict) -> str:
    directions = discover.get("top_directions") or []
    if directions:
        return str(directions[0].get("name", "OpenSourcePR contribution"))
    return "OpenSourcePR contribution"


def _workflow_title(selected: str, changed_files: list[str]) -> str:
    scope = "docs" if changed_files and all(_is_doc_file(path) for path in changed_files) else "agent"
    text = re.sub(r"[^A-Za-z0-9一-龥]+", " ", selected).strip()
    words = " ".join(text.split()[:8]) or "update contribution workflow"
    return f"feat({scope}): {words}"


def _is_doc_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".md", ".rst", ".txt")) or lower.startswith("docs/")


def _extract_section(markdown: str, heading: str) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", re.M | re.S)
    match = pattern.search(markdown)
    return match.group(1).strip() if match else ""


def _reviewer_notes(design: dict, implementation: str) -> str:
    notes = [
        "Review whether the implementation remains within the selected OpenSourcePR scope.",
        "Check that the code changes match the saved technical design artifact.",
    ]
    if design.get("agent_design"):
        notes.append("The design was refined by an agent review artifact; compare the implementation against that section.")
    if "No explicit test" in implementation:
        notes.append("Testing evidence is incomplete and should be filled before opening the PR.")
    return "\n".join(f"- {note}" for note in notes)
