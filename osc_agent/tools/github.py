from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from http.client import IncompleteRead
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import Field

from osc_agent.runtime.models import (
    ContractModel,
    ToolError,
    ToolResult,
    ToolUseContext,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool


class GitHubIssue(ContractModel):
    number: int = Field(ge=1)
    title: str
    body: str
    state: str
    url: str
    labels: list[str]
    updated_at: str


class GitHubListIssuesInput(ContractModel):
    repo_url: str = Field(min_length=1, description="Public GitHub repository URL.")
    labels: list[str] = Field(default_factory=list, description="Optional case-insensitive labels.")
    updated_within_days: int = Field(default=60, ge=1, le=365)


class GitHubListIssuesOutput(ContractModel):
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
        return _validate_repo_url(input.repo_url)

    async def call(self, input: GitHubListIssuesInput, context: ToolUseContext) -> ToolResult:
        result = await asyncio.to_thread(
            fetch_issues,
            input.repo_url,
            labels=input.labels,
            updated_within_days=input.updated_within_days,
        )
        if not result.get("ok"):
            return _github_tool_error(result)
        return ToolResult(data={"issues": [_normalize_issue(issue) for issue in result["issues"]]})


class GitHubGetIssueInput(ContractModel):
    repo_url: str = Field(min_length=1, description="Public GitHub repository URL.")
    issue_number: int = Field(ge=1)


class GitHubGetIssueOutput(ContractModel):
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
        return _validate_repo_url(input.repo_url)

    async def call(self, input: GitHubGetIssueInput, context: ToolUseContext) -> ToolResult:
        result = await asyncio.to_thread(fetch_issue, input.repo_url, input.issue_number)
        if not result.get("ok"):
            return _github_tool_error(result)
        return ToolResult(data={"issue": _normalize_issue(result["issue"])})


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


def fetch_issue(repo_url: str, issue_number: int, token: str | None = None) -> dict[str, Any]:
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
