"""提供只读 GitHub Issue 查询工具。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import Field

from osc_agent.runtime.models import (
    ContractModel,
    ToolError,
    ToolResult,
    ToolUseContext,
    ToolResultBlock,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.tools.process_runner import build_subprocess_environment


class GitHubIssue(ContractModel):
    number: int = Field(ge=1)
    title: str
    body: str
    state: str
    url: str
    labels: list[str]
    updated_at: str
    comments: list["GitHubComment"] = Field(default_factory=list)


class GitHubComment(ContractModel):
    author: str
    body: str
    created_at: str
    url: str


class GitHubListIssuesInput(ContractModel):
    repo_url: str = Field(min_length=1, description="Public GitHub repository URL.")
    labels: list[str] = Field(default_factory=list, description="Optional case-insensitive labels.")
    updated_within_days: int = Field(default=60, ge=1, le=365)


class GitHubListIssuesOutput(ContractModel):
    content_source: Literal["github"] = "github"
    trust: Literal["untrusted_external"] = "untrusted_external"
    issues: list[GitHubIssue]


class GitHubListIssuesTool(BaseTool[GitHubListIssuesInput, GitHubListIssuesOutput]):
    name = "github_list_issues"
    description = "Read a bounded list of issues from a public GitHub repository."
    input_model = GitHubListIssuesInput
    output_model = GitHubListIssuesOutput

    def is_read_only(self, input: GitHubListIssuesInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitHubListIssuesInput) -> bool:
        return True

    async def validate_input(self, input: GitHubListIssuesInput, context: ToolUseContext) -> ValidationResult:
        return _validate_remote(input.repo_url, context)

    async def call(self, input: GitHubListIssuesInput, context: ToolUseContext) -> ToolResult:
        result = await asyncio.to_thread(
            fetch_issues,
            input.repo_url,
            labels=input.labels,
            updated_within_days=input.updated_within_days,
        )
        if not result.get("ok"):
            return _github_tool_error(result)
        return ToolResult(
            data={
                "content_source": "github",
                "trust": "untrusted_external",
                "issues": [_normalize_issue(issue) for issue in result["issues"]],
            }
        )


class GitHubGetIssueInput(ContractModel):
    repo_url: str = Field(min_length=1, description="Public GitHub repository URL.")
    issue_number: int = Field(ge=1)
    max_comments: int = Field(default=30, ge=0, le=100)


class GitHubGetIssueOutput(ContractModel):
    content_source: Literal["github"] = "github"
    trust: Literal["untrusted_external"] = "untrusted_external"
    issue: GitHubIssue


class GitHubGetIssueTool(BaseTool[GitHubGetIssueInput, GitHubGetIssueOutput]):
    name = "github_get_issue"
    description = "Read one issue and its metadata from a public GitHub repository."
    input_model = GitHubGetIssueInput
    output_model = GitHubGetIssueOutput

    def is_read_only(self, input: GitHubGetIssueInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitHubGetIssueInput) -> bool:
        return True

    async def validate_input(self, input: GitHubGetIssueInput, context: ToolUseContext) -> ValidationResult:
        valid = _validate_remote(input.repo_url, context)
        return valid

    async def call(self, input: GitHubGetIssueInput, context: ToolUseContext) -> ToolResult:
        result = await asyncio.to_thread(
            fetch_issue,
            input.repo_url,
            input.issue_number,
            max_comments=input.max_comments,
        )
        if not result.get("ok"):
            return _github_tool_error(result)
        return ToolResult(
            data={
                "content_source": "github",
                "trust": "untrusted_external",
                "issue": _normalize_issue(result["issue"]),
            }
        )


def _validate_remote(repo_url: str, context: ToolUseContext) -> ValidationResult:
    valid = _validate_repo_url(repo_url)
    if isinstance(valid, ValidationFailure):
        return valid
    remote = _local_origin(Path(context.working_directory))
    requested = parse_github_repo(repo_url)
    if (
        remote is not None
        and tuple(part.casefold() for part in remote)
        != tuple(part.casefold() for part in requested)
        and not _remote_mismatch_approved(
            context,
            local_origin=remote,
            requested_repository=requested,
        )
    ):
        return ValidationFailure(
            reason=(
                f"repo_url does not match local origin {remote[0]}/{remote[1]}; "
                "ask the user with purpose=remote_mismatch and bind the exact repository pair"
            )
        )
    return ValidationSuccess()


def _validate_repo_url(repo_url: str) -> ValidationResult:
    try:
        parse_github_repo(repo_url)
    except ValueError as exc:
        return ValidationFailure(reason=str(exc))
    return ValidationSuccess()


def _normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue.get("number"),
        "title": str(issue.get("title") or ""),
        "body": str(issue.get("body") or ""),
        "state": str(issue.get("state") or ""),
        "url": str(issue.get("html_url") or ""),
        "labels": sorted(_issue_labels(issue)),
        "updated_at": str(issue.get("updated_at") or ""),
        "comments": [
            {
                "author": str((comment.get("user") or {}).get("login") or ""),
                "body": str(comment.get("body") or ""),
                "created_at": str(comment.get("created_at") or ""),
                "url": str(comment.get("html_url") or ""),
            }
            for comment in issue.get("_comments") or []
        ],
    }


def _github_tool_error(result: dict[str, Any]) -> ToolResult:
    return ToolResult(
        error=ToolError(
            code="GITHUB_READ_FAILED",
            message=str(result.get("error") or "GitHub read failed"),
            retryable=True,
        )
    )


def parse_github_repo(repo_url: str) -> tuple[str, str]:
    parsed = urlparse(repo_url.strip())
    if parsed.netloc not in {"github.com", "www.github.com"}:
        raise ValueError("repo_url must be a GitHub repository URL")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("repo_url must contain exactly an owner and repository name")
    return parts[0], parts[1].removesuffix(".git")


def fetch_issues(
    repo_url: str,
    labels: list[str] | None = None,
    updated_within_days: int = 60,
    token: str | None = None,
) -> dict[str, Any]:
    try:
        owner, repo = parse_github_repo(repo_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "issues": []}
    since = datetime.now(timezone.utc) - timedelta(days=max(updated_within_days, 1))
    query = f"?state=open&per_page=100&since={since.isoformat().replace('+00:00', 'Z')}"
    result = _github_get_json(f"https://api.github.com/repos/{owner}/{repo}/issues{query}", token)
    if not result["ok"]:
        return {**result, "issues": []}
    issues = [item for item in result["data"] if "pull_request" not in item]
    if labels:
        wanted = {label.casefold() for label in labels}
        issues = [issue for issue in issues if _issue_labels(issue) & wanted]
    return {"ok": True, "issues": issues}


def fetch_issue(
    repo_url: str,
    issue_number: int,
    token: str | None = None,
    *,
    max_comments: int = 30,
) -> dict[str, Any]:
    try:
        owner, repo = parse_github_repo(repo_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "issue": {}}
    result = _github_get_json(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
        token,
    )
    if not result["ok"]:
        return {**result, "issue": {}}
    issue = result.get("data") or {}
    if issue.get("pull_request"):
        return {"ok": False, "error": "requested number is a pull request", "issue": {}}
    if max_comments:
        comments = _github_get_json(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page={max_comments}",
            token,
        )
        if not comments["ok"]:
            return {"ok": False, "error": comments["error"], "issue": {}}
        issue["_comments"] = list(comments.get("data") or [])[:max_comments]
    return {"ok": True, "issue": issue}


def _github_get_json(url: str, token: str | None = None) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "open-source-contribution-agent",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = token or os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=20) as response:
            return {"ok": True, "data": json.loads(response.read().decode("utf-8"))}
    except HTTPError as exc:
        return {"ok": False, "error": f"GitHub API returned HTTP {exc.code}"}
    except IncompleteRead:
        return {"ok": False, "error": "GitHub API returned an incomplete response"}
    except (URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": f"GitHub connection failed: {exc}"}


def _issue_labels(issue: dict[str, Any]) -> set[str]:
    return {
        str(label.get("name", "") if isinstance(label, dict) else label).casefold()
        for label in issue.get("labels") or []
    }


def _local_origin(repo_root: Path) -> tuple[str, str] | None:
    try:
        top = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_root.resolve()}",
                "rev-parse",
                "--show-toplevel",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=build_subprocess_environment(),
        )
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != repo_root.resolve():
            return None
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_root.resolve()}",
                "config",
                "--get",
                "remote.origin.url",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=build_subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        return None
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    elif value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix("ssh://git@github.com/")
    try:
        return parse_github_repo(value)
    except ValueError:
        return None


def _remote_mismatch_approved(
    context: ToolUseContext,
    *,
    local_origin: tuple[str, str],
    requested_repository: tuple[str, str],
) -> bool:
    expected_local = "/".join(part.casefold() for part in local_origin)
    expected_requested = "/".join(part.casefold() for part in requested_repository)
    for message in context.transcript_messages:
        for block in message.content:
            if not isinstance(block, ToolResultBlock) or not isinstance(block.content, dict):
                continue
            data = block.content.get("data")
            if not isinstance(data, dict):
                continue
            questions = data.get("questions")
            answers = data.get("answers")
            if not isinstance(questions, list) or not isinstance(answers, list):
                continue
            approved_ids = {
                str(answer.get("question_id"))
                for answer in answers
                if isinstance(answer, dict)
                and answer.get("selected_option_id") == "proceed"
            }
            if any(
                isinstance(question, dict)
                and question.get("purpose") == "remote_mismatch"
                and str(question.get("id")) in approved_ids
                and isinstance(question.get("remote_mismatch_reference"), dict)
                and str(
                    question["remote_mismatch_reference"].get("local_origin") or ""
                ).casefold()
                == expected_local
                and str(
                    question["remote_mismatch_reference"].get("requested_repository") or ""
                ).casefold()
                == expected_requested
                for question in questions
            ):
                return True
    return False
