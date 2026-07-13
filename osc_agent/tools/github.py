"""
输入 GitHub 仓库地址
      ↓
parse_github_repo()
解析 owner / repo
      ↓
fetch_issues()
调用 GitHub Issues API
      ↓
fetch_issue_comments()
读取 Issue 评论
      ↓
filter_candidate_issues()
筛选适合贡献的 Issue
      ↓
返回候选 Issue 列表
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from http.client import IncompleteRead
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

CANDIDATE_LABELS = {"good first issue", "help wanted", "bug", "enhancement"}
CLAIM_PATTERNS = (
    "i'll take this",
    "i will take this",
    "i’m working on this",
    "i'm working on this",
    "i am working on this",
)


def parse_github_repo(repo_url: str) -> tuple[str, str]:
    """从 GitHub URL 中提取 owner/repo，避免后续 API 路径拼接时靠字符串猜测。"""
    parsed = urlparse(repo_url.strip())
    if parsed.netloc not in {"github.com", "www.github.com"}:
        raise ValueError("repo_url must be a GitHub repository URL")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("repo_url must include owner and repository name")
    repo = parts[1].removesuffix(".git")
    return parts[0], repo


def fetch_issues(repo_url: str, labels: list[str] | None = None, updated_within_days: int = 60, token: str | None = None) -> dict[str, Any]:
    """只读拉取 GitHub issues；失败时返回结构化错误，调用方可改用 --issues-file。

    Args:
        repo_url: GitHub 仓库 URL
        labels: 要筛选的标签列表
        updated_within_days: 获取最近 N 天内更新的 issues
        token: GitHub 访问令牌（可选，未提供则从环境变量读取）
    """
    try:
        owner, repo = parse_github_repo(repo_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "issues": []}

    since = datetime.now(timezone.utc) - timedelta(days=max(int(updated_within_days), 1))
    query = f"?state=open&per_page=100&since={since.isoformat().replace('+00:00', 'Z')}"
    result = _github_get_json(f"https://api.github.com/repos/{owner}/{repo}/issues{query}", token=token)
    if not result["ok"]:
        return {**result, "issues": []}

    issues = [item for item in result["data"] if "pull_request" not in item]
    if labels:
        wanted = {label.lower() for label in labels}
        issues = [issue for issue in issues if _issue_labels(issue) & wanted]
    return {"ok": True, "issues": issues}


def fetch_issue_comments(repo_url: str, issue_number: int, token: str | None = None) -> dict[str, Any]:
    """只读拉取单个 issue 评论，用于判断是否已有贡献者认领。"""
    try:
        owner, repo = parse_github_repo(repo_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "comments": []}

    result = _github_get_json(f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=100", token=token)
    if not result["ok"]:
        return {**result, "comments": []}
    return {"ok": True, "comments": result["data"]}


def fetch_issue(repo_url: str, issue_number: int, token: str | None = None) -> dict[str, Any]:
    """只读获取单个 Issue，供长流程在进入实现前重新确认远端状态。"""
    try:
        owner, repo = parse_github_repo(repo_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "issue": {}}

    result = _github_get_json(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
        token=token,
    )
    if not result["ok"]:
        return {**result, "issue": {}}
    issue = result.get("data") or {}
    if issue.get("pull_request"):
        return {"ok": False, "error": "requested number is a pull request", "issue": {}}
    return {"ok": True, "issue": issue}


def fetch_issue_activity(repo_url: str, issue_number: int, token: str | None = None) -> dict[str, Any]:
    """只读检查 Issue 时间线中的关联 PR，并附带仓库近期提交摘要。"""
    try:
        owner, repo = parse_github_repo(repo_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "linked_pull_requests": [], "recent_commits": []}
    timeline = _github_get_json(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/timeline?per_page=100",
        token=token
    )
    commits = _github_get_json(f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=20", token=token)
    linked: list[dict[str, Any]] = []
    if timeline.get("ok"):
        for event in timeline.get("data") or []:
            source = event.get("source") or {}
            issue = source.get("issue") or {}
            if issue.get("pull_request"):
                linked.append(
                    {
                        "number": issue.get("number"),
                        "state": issue.get("state"),
                        "url": issue.get("html_url"),
                        "title": issue.get("title"),
                    }
                )
    recent = []
    if commits.get("ok"):
        for item in commits.get("data") or []:
            commit = item.get("commit") or {}
            recent.append(
                {
                    "sha": item.get("sha"),
                    "message": str((commit.get("message") or "").splitlines()[0]),
                    "url": item.get("html_url"),
                }
            )
    return {
        "ok": bool(timeline.get("ok") and commits.get("ok")),
        "linked_pull_requests": linked,
        "recent_commits": recent,
    }


def filter_candidate_issues(
    issues: list[dict[str, Any]],
    comments_by_issue: dict[int | str, list[dict[str, Any]]],
    *,
    updated_within_days: int = 60,
) -> list[dict[str, Any]]:
    """按 OpenSourcePR 第一步规则筛选候选 issue，并保留推荐理由所需字段。"""
    threshold = datetime.now(timezone.utc) - timedelta(days=max(int(updated_within_days), 1))
    candidates: list[dict[str, Any]] = []
    for issue in issues:
        if issue.get("state") != "open":
            continue
        if not (_issue_labels(issue) & CANDIDATE_LABELS):
            continue
        if issue.get("assignee") or issue.get("assignees"):
            continue
        if _parse_github_time(str(issue.get("updated_at", ""))) < threshold:
            continue
        if not _has_clear_description(str(issue.get("body") or "")):
            continue
        issue_number = issue.get("number")
        comments = comments_by_issue.get(issue_number, comments_by_issue.get(str(issue_number), []))
        if _has_claim_comment(comments):
            continue
        linked_pull_requests = (issue.get("activity") or {}).get("linked_pull_requests") or []
        if any(str(item.get("state", "")).lower() == "open" for item in linked_pull_requests):
            continue
        candidates.append(
            {
                "number": issue_number,
                "title": issue.get("title", ""),
                "labels": sorted(_issue_labels(issue)),
                "url": issue.get("html_url", ""),
                "updated_at": issue.get("updated_at", ""),
                "body": issue.get("body") or "",
                "activity": issue.get("activity") or {},
            }
        )
    return candidates


def apply_issue_scores(
    candidates: list[dict[str, Any]],
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并可解释等级；未标定数据前不输出伪精确的百分制分数。"""
    rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "REJECT": 0}
    by_number = {score.get("number"): score for score in scores if isinstance(score, dict)}
    ranked: list[dict[str, Any]] = []
    for issue in candidates:
        score = by_number.get(issue.get("number"))
        if score:
            level = str(score.get("level") or "REJECT").upper()
            issue = {
                **issue,
                "review_level": level,
                "review_dimensions": score.get("dimensions") or {},
                "review_reason": score.get("reason", ""),
                "rejection_reason": score.get("rejection_reason", ""),
            }
            if level in {"LOW", "REJECT"}:
                continue
        ranked.append(issue)
    return sorted(ranked, key=lambda item: rank.get(str(item.get("review_level", "MEDIUM")), 2), reverse=True)


def load_issues_file(path: str) -> tuple[list[dict[str, Any]], dict[int | str, list[dict[str, Any]]]]:
    """读取离线 issue 文件；支持纯数组或包含 issues/comments_by_issue 的对象。"""
    text = open(path, encoding="utf-8").read()
    data = json.loads(text)
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict):
        issues = data.get("issues", [])
        comments = data.get("comments_by_issue", {})
        if isinstance(issues, list) and isinstance(comments, dict):
            return issues, comments
    raise ValueError("issues file must be a JSON list or an object with issues/comments_by_issue")


def _github_get_json(url: str, token: str | None = None) -> dict[str, Any]:
    """调用 GitHub API，支持传入 token 或从环境变量读取。"""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "open-source-contribution-agent",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # 优先使用传入的 token，否则从环境变量读取
    github_token = token or os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            return {"ok": True, "data": json.loads(response.read().decode("utf-8"))}
    except HTTPError as exc:
        error_msg = f"GitHub API error: HTTP {exc.code}"
        if exc.code == 401:
            error_msg += " (Unauthorized - check your GITHUB_TOKEN)"
        elif exc.code == 403:
            error_msg += " (Forbidden - likely rate limit exceeded without token)"
        elif exc.code == 404:
            error_msg += " (Not Found - check repository URL)"
        return {
            "ok": False,
            "error": f"{error_msg}. Use --issues-file for offline input.",
        }
    except IncompleteRead as exc:
        return {
            "ok": False,
            "error": f"GitHub API incomplete response (network issue). Retry or use --issues-file.",
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "error": f"GitHub connection failed: {exc}. Check network or use --issues-file.",
        }


def _issue_labels(issue: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for label in issue.get("labels") or []:
        if isinstance(label, dict):
            labels.add(str(label.get("name", "")).lower())
        else:
            labels.add(str(label).lower())
    return labels


def _parse_github_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _has_clear_description(body: str) -> bool:
    if len(body.strip()) < 40:
        return False
    return bool(re.search(r"(expected|actual|steps|reproduce|should|error|bug|feature|期望|复现|步骤)", body, re.I))


def _has_claim_comment(comments: list[dict[str, Any]]) -> bool:
    for comment in comments:
        body = str(comment.get("body") or "").lower()
        if any(pattern in body for pattern in CLAIM_PATTERNS):
            return True
    return False
