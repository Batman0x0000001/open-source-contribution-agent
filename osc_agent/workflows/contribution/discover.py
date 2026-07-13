from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
from typing import Any

from osc_agent.harness.contracts import RunStatus
from osc_agent.tools.github import (
    CANDIDATE_LABELS,
    apply_issue_scores,
    fetch_issue,
    fetch_issue_activity,
    fetch_issue_comments,
    fetch_issues,
    filter_candidate_issues,
    load_issues_file,
)
from osc_agent.tools.repo import (
    analyze_architecture_dimensions,
    collect_repo_evidence_pack,
    detect_entrypoints,
    inspect_repo,
    repo_tree,
)
from osc_agent.workflows.contribution.gates import GateResult
from osc_agent.workflows.contribution.models import ContributionRun
from osc_agent.workflows.contribution.state import (
    _evidence_file_hashes,
    _read_json,
    _require_consistent_run,
    _write_json,
    _write_metrics_report,
    _write_text,
    create_run,
    load_run,
    save_run,
)
from osc_agent.workflows.contribution.transitions import _begin_stage, _complete_stage

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
    _begin_stage(run, "discover", repo_root)

    # 从 settings 获取 GitHub token，避免未授权 API 访问受限
    github_token = getattr(settings, "github_token", None) if settings else None

    issues, comments_by_issue, issue_error = _collect_issues(
        repo_url=repo_url,
        issues_file=issues_file,
        github_token=github_token
    )
    if issue_error:
        print(f"⚠️  Warning: {issue_error}")
        if not github_token and not os.getenv("GITHUB_TOKEN"):
            print("💡 Tip: Set GITHUB_TOKEN in your .env file for authenticated GitHub access")
            print("   Without token: 60 requests/hour | With token: 5,000 requests/hour")
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
        "issue_source": "offline" if issues_file is not None else "github",
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


def build_discover_evidence(*, repo_root: Path) -> dict[str, Any]:
    return collect_repo_evidence_pack(repo_root=repo_root)


def revalidate_selected_issue(run: ContributionRun) -> GateResult:
    """进入实现前重新确认选中 Issue 仍开放且没有被认领。"""
    discover = _read_json(run, "01_discover.json")
    if discover.get("issue_source") == "offline":
        return GateResult(True, "offline issue snapshot cannot be refreshed")

    selected = run.selected_direction or ""
    match = re.search(r"Issue\s+#(\d+)", selected, flags=re.IGNORECASE)
    if not match:
        return GateResult(True, "selected direction is not tied to a GitHub issue")
    number = int(match.group(1))
    current = fetch_issue(run.repo_url, number)
    if not current.get("ok"):
        return GateResult(
            False,
            f"could not revalidate Issue #{number}: {current.get('error', 'unknown error')}",
            status=RunStatus.BLOCKED_NEEDS_USER,
        )
    issue = current.get("issue") or {}
    comments = fetch_issue_comments(run.repo_url, number)
    activity = fetch_issue_activity(run.repo_url, number)
    issue["activity"] = activity if activity.get("ok") else {}
    candidates = filter_candidate_issues(
        [issue],
        {number: comments.get("comments") or []},
        updated_within_days=3650,
    )
    if not candidates:
        return GateResult(
            False,
            f"Issue #{number} is closed, assigned, claimed, or already has an active linked PR",
            status=RunStatus.BLOCKED_NEEDS_USER,
        )
    run.issue_snapshot_at = datetime.now(timezone.utc).isoformat()
    if run.metrics is not None:
        run.metrics["issue_revalidated_at"] = run.issue_snapshot_at
    save_run(run)
    return GateResult(True, f"Issue #{number} is still available")


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


def render_discover(payload: dict[str, Any]) -> str:
    issue_error = payload.get("issue_error")
    if issue_error:
        issue_error_section = (
            f"\n\n**⚠️ GitHub API Error:**\n\n"
            f"```\n{issue_error}\n```\n\n"
            f"**Troubleshooting:**\n"
            f"- Set `GITHUB_TOKEN` in your `.env` file for authenticated access\n"
            f"- Use `--issues-file <path>` to provide issues offline\n"
            f"- Check your network connection and GitHub API rate limits\n\n"
        )
    else:
        issue_error_section = ""

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
        f"{issue_error_section}"
        "| Issue | Title | Labels |\n|---|---|---|\n"
        f"{issue_rows}\n\n"
        "## Architecture Gap Analysis\n"
        f"{dimensions}\n"
        f"{review}\n"
        "## Top 3 Contribution Suggestions\n\n"
        f"{directions}\n"
    )


def _collect_issues(
    *,
    repo_url: str,
    issues_file: Path | None,
    github_token: str | None = None,
) -> tuple[list[dict[str, Any]], dict[int | str, list[dict[str, Any]]], str | None]:
    """收集 GitHub issues，支持传入 token 避免未授权 API 访问受限。"""
    if issues_file is not None:
        issues, comments = load_issues_file(str(issues_file))
        return issues, comments, None

    # 使用 token 调用 GitHub API
    issue_result = fetch_issues(repo_url, labels=sorted(CANDIDATE_LABELS), token=github_token)
    if not issue_result["ok"]:
        return [], {}, issue_result["error"]

    comments: dict[int | str, list[dict[str, Any]]] = {}
    for issue in issue_result["issues"][:20]:
        number = issue.get("number")
        if number is None:
            continue
        result = fetch_issue_comments(repo_url, int(number), token=github_token)
        comments[number] = result.get("comments", [])
        issue["activity"] = fetch_issue_activity(repo_url, int(number), token=github_token)
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


def _normalize_directions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and item.get("name") and item.get("description")]


def _try_import_stage_agents():
    try:
        from osc_agent.workflows.contribution import agents as stage_agents

        return stage_agents
    except ImportError:
        return None
