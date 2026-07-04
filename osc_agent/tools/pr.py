from __future__ import annotations

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


def draft_pr(*, repo_root: Path) -> str:
    """基于当前本地 diff 生成 PR 草稿；只读 git 状态，不提交、不推送、不创建 PR。"""
    diff = git_diff(repo_root=repo_root)
    status = git_status(repo_root=repo_root)
    draft = build_pr_draft(diff=diff, status=status)
    return format_pr_draft(draft)


def build_pr_draft(*, diff: str, status: str) -> PRDraft:
    """把 git diff/status 提炼成稳定结构，便于 CLI 和测试复用同一套 PR 草稿逻辑。"""
    changed_files = _changed_files(diff, status)
    title = _title_for_files(changed_files)
    summary = _summary_for_files(changed_files)
    return PRDraft(
        title=title,
        summary=summary,
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
