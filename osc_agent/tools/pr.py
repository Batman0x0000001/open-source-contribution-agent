from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from osc_agent.tools.git import git_diff, git_status

PR_TOOLS = [
    {
        "name": "draft_pr",
        "description": "Draft a local pull request title and body from the current git diff.",
        "input_schema": {"type": "object", "properties": {}},
    }
]


@dataclass(frozen=True)
class PRDraft:
    title: str
    summary: list[str]
    tests: list[str]
    risk: str


def draft_pr(*, repo_root: Path, run_id: str | None = None) -> str:
    """生成 PR 草稿；传入 run_id 时读取工作流上下文，但始终不提交、不推送、不创建 PR。"""
    if run_id:
        return _draft_from_workflow(repo_root=repo_root, run_id=run_id)
    diff = git_diff(repo_root=repo_root)
    status = git_status(repo_root=repo_root)
    return format_pr_draft(build_pr_draft(diff=diff, status=status))


def build_pr_draft(*, diff: str, status: str) -> PRDraft:
    """把 git diff/status 提炼成稳定结构，便于 CLI 和测试复用同一套 PR 草稿逻辑。"""
    changed_files = _changed_files(diff, status)
    return PRDraft(
        title=_title_for_files(changed_files),
        summary=_summary_for_files(changed_files),
        tests=["Not run (not provided)."],
        risk=_risk_for_files(changed_files),
    )


def format_pr_draft(draft: PRDraft) -> str:
    """输出可直接复制到 PR 描述里的 Markdown 文本。"""
    summary = "\n".join(f"- {item}" for item in draft.summary)
    tests = "\n".join(f"- {item}" for item in draft.tests)
    return (
        f"Title: {draft.title}\n\n"
        "## Summary\n"
        f"{summary}\n\n"
        "## Tests\n"
        f"{tests}\n\n"
        "## Risk\n"
        f"- {draft.risk}"
    )


def _draft_from_workflow(*, repo_root: Path, run_id: str) -> str:
    artifacts_dir = repo_root / ".osc_agent" / "contribution_runs" / run_id
    discover = _read_artifact_json(artifacts_dir / "01_discover.json")
    design = _read_artifact_json(artifacts_dir / "02_design.json")
    implementation = _read_artifact_text(artifacts_dir / "03_implementation_report.md")
    changed_files = _changed_files(git_diff(repo_root=repo_root), git_status(repo_root=repo_root))
    selected = str(design.get("selected_direction") or _first_direction_name(discover))
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


def _title_for_files(files: list[str]) -> str:
    if not files:
        return "Draft PR: no local changes"
    if all(_is_doc_file(path) for path in files):
        return "Update documentation"
    if all(path.startswith("tests/") or path.startswith("test/") for path in files):
        return "Update tests"
    return "Update contribution files"


def _summary_for_files(files: list[str]) -> list[str]:
    if not files:
        return ["No local changes detected."]
    preview = ", ".join(files[:5])
    if len(files) > 5:
        preview += f", and {len(files) - 5} more"
    return [f"Updates {preview}."]


def _risk_for_files(files: list[str]) -> str:
    if not files:
        return "No code or documentation changes detected."
    if all(_is_doc_file(path) for path in files):
        return "Low; documentation-only change."
    return "Review the diff and run the relevant project tests before opening a PR."


def _is_doc_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith((".md", ".rst", ".txt")) or lower.startswith("docs/")


def _read_artifact_json(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"required workflow artifact missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_artifact_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _first_direction_name(discover: dict) -> str:
    directions = discover.get("top_directions") or []
    if directions:
        return str(directions[0].get("name", "OpenSourcePR contribution"))
    return "OpenSourcePR contribution"


def _workflow_title(selected: str, changed_files: list[str]) -> str:
    scope = "docs" if changed_files and all(_is_doc_file(path) for path in changed_files) else "agent"
    text = re.sub(r"[^A-Za-z0-9一-龥 ]+", " ", selected).strip()
    words = " ".join(text.split()[:8]) or "update contribution workflow"
    return f"feat({scope}): {words}"


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
