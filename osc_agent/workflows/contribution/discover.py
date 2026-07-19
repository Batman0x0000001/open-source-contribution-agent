from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from osc_agent.harness.contracts import RunStatus
from osc_agent.workflows.contribution.agents import run_discover_analysis, score_candidate_issues
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
    analyze_issue_code_candidates,
    collect_repo_evidence_pack,
    detect_entrypoints,
    inspect_repo,
    repo_tree,
)
from osc_agent.workflows.contribution.gates import GateResult
from osc_agent.workflows.contribution.models import ContributionDirection, ContributionRun
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
    client: Any,
    settings: Any,
    issues_file: Path | None = None,
) -> ContributionRun:
    run = create_run(repo_root=repo_root, repo_url=repo_url, settings=settings)
    _begin_stage(run, "discover", repo_root)
    try:
        github_token = getattr(settings, "github_token", None)
        issues, comments_by_issue, issue_error = _collect_issues(
            repo_url=repo_url,
            issues_file=issues_file,
            github_token=github_token,
        )
        collection_warnings = [issue_error] if issue_error else []
        for issue in issues:
            number = issue.get("number", "unknown")
            if issue.get("comments_error"):
                collection_warnings.append(f"Issue #{number} comments: {issue['comments_error']}")
            if issue.get("activity_error"):
                collection_warnings.append(f"Issue #{number} activity: {issue['activity_error']}")
        candidates = filter_candidate_issues(issues, comments_by_issue)
        for issue in candidates:
            issue["code_candidates"] = analyze_issue_code_candidates(repo_root=repo_root, issue=issue)

        # 只生成一次仓库快照，保证 LLM 输入和落盘 Artifact 使用相同事实。
        repo_overview = inspect_repo(repo_root=repo_root)
        tree = repo_tree(repo_root=repo_root, depth=3)
        entrypoints = detect_entrypoints(repo_root=repo_root)
        dimensions = analyze_architecture_dimensions(repo_root=repo_root)
        evidence_pack = build_discover_evidence(repo_root=repo_root)
        issue_scores = score_candidate_issues(
            client,
            settings,
            candidates,
            comments_by_issue,
            repo_root=repo_root,
        )
        candidates = apply_issue_scores(candidates, issue_scores)
        llm_result = run_discover_analysis(
            client,
            settings,
            {
                "repo_url": repo_url,
                "repo_overview": repo_overview,
                "tree": tree,
                "entrypoints": entrypoints,
                "candidate_issues": candidates,
                "issue_scores": issue_scores,
                "architecture_dimensions": dimensions,
                "evidence_pack": evidence_pack,
            },
            repo_root=repo_root,
        )
        directions = _normalize_directions(llm_result.get("top_directions"), candidates)
        if not directions:
            raise ValueError("Discover model returned no valid contribution directions")
        analysis_summary = str(llm_result.get("analysis_summary") or "").strip()
        if not analysis_summary:
            raise ValueError("Discover model returned an empty analysis_summary")

        payload = {
            "repo_url": repo_url,
            "repo_overview": repo_overview,
            "tree": tree,
            "entrypoints": entrypoints,
            "candidate_issues": candidates,
            "issue_scores": issue_scores,
            "architecture_dimensions": list(llm_result.get("architecture_insights") or []),
            "top_directions": directions,
            "warnings": collection_warnings,
            "issue_source": "offline" if issues_file is not None else "github",
            "evidence_pack": evidence_pack,
            "repository_profile": evidence_pack.get("repository_profile", {}),
            "llm_analysis_summary": analysis_summary,
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
        _write_metrics_report(run)
        return run
    except Exception as exc:
        run.final_status = (
            RunStatus.FAILED_VALIDATION.value if isinstance(exc, ValueError) else RunStatus.FAILED_TOOL.value
        )
        if run.metrics is not None:
            run.metrics["failure_reason"] = str(exc)[:1000]
        if (run.stage_status or {}).get("discover") == "RUNNING":
            _complete_stage(run, "discover", success=False)
        _write_metrics_report(run)
        raise


def attach_discover_human_review(*, repo_root: Path, run_id: str, review: str) -> ContributionRun:
    if not review.strip():
        raise ValueError("human review is required")
    run = load_run(repo_root=repo_root, run_id=run_id)
    _require_consistent_run(run, repo_root)
    if run.stage_status.get("discover") != "SUCCEEDED":
        raise ValueError("Discover must succeed before attaching a human review")
    downstream = ("design", "implement", "draft_pr")
    if any(run.stage_status.get(stage) != "PENDING" for stage in downstream):
        raise ValueError("human review cannot change after a downstream stage has started")
    _write_json(
        run,
        "01_discover_human_review.json",
        {
            "review": review.strip(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    save_run(run)
    return run


def build_discover_evidence(*, repo_root: Path) -> dict[str, Any]:
    return collect_repo_evidence_pack(repo_root=repo_root)


def revalidate_selected_issue(run: ContributionRun, *, github_token: str | None = None) -> GateResult:
    """进入实现前重新确认选中 Issue 仍开放且没有被认领。"""
    discover = _read_json(run, "01_discover.json")
    if discover.get("issue_source") == "offline":
        return GateResult(True, "offline issue snapshot cannot be refreshed")

    number = run.selected_issue_number
    if number is None:
        return GateResult(True, "selected direction is not tied to a GitHub issue")
    current = fetch_issue(run.repo_url, number, token=github_token)
    if not current.get("ok"):
        return GateResult(
            False,
            f"could not revalidate Issue #{number}: {current.get('error', 'unknown error')}",
            status=RunStatus.BLOCKED_NEEDS_USER,
        )
    issue = current.get("issue") or {}
    comments = fetch_issue_comments(run.repo_url, number, token=github_token)
    activity = fetch_issue_activity(run.repo_url, number, token=github_token)
    if not comments.get("ok") or not activity.get("ok"):
        failures = [
            str(result.get("error") or label)
            for label, result in (("comments unavailable", comments), ("activity unavailable", activity))
            if not result.get("ok")
        ]
        return GateResult(
            False,
            f"could not fully revalidate Issue #{number}: {'; '.join(failures)}",
            status=RunStatus.BLOCKED_NEEDS_USER,
        )
    issue["activity"] = activity
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
    checked_at = datetime.now(timezone.utc).isoformat()
    return GateResult(
        True,
        f"Issue #{number} is still available",
        metadata={"issue_revalidated_at": checked_at},
    )


def build_discover_review_prompt(
    *,
    repo_url: str,
    candidates: list[dict[str, Any]],
    dimensions: list[dict[str, str]],
    evidence_pack: dict[str, Any],
) -> str:
    prompt = (
        "OpenSourcePR step 1: find contribution entry points.\n"
        f"Repository: {repo_url}\n\n"
        f"Candidate issues:\n{json.dumps(candidates, ensure_ascii=False, indent=2)[:12000]}\n\n"
        f"Architecture dimensions:\n{json.dumps(dimensions, ensure_ascii=False, indent=2)[:8000]}\n\n"
        f"Evidence pack:\n{json.dumps(evidence_pack, ensure_ascii=False, indent=2)[:10000]}\n"
    )
    return prompt[:32_000]


def render_discover(payload: dict[str, Any]) -> str:
    warnings = [str(item) for item in payload.get("warnings") or []]
    if warnings:
        issue_error_section = (
            "\n\n**Collection warnings:**\n\n"
            + "\n".join(f"- {_markdown_cell(item)}" for item in warnings)
            + "\n\n"
        )
    else:
        issue_error_section = ""

    issue_rows = "\n".join(
        f"| #{issue.get('number')} | {_markdown_cell(issue.get('title', ''))} | "
        f"{_markdown_cell(', '.join(issue.get('labels', [])))} |"
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
    analysis = payload.get("llm_analysis_summary") or ""
    review = f"\n## LLM Analysis\n\n{analysis}\n" if analysis else ""
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


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


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

    selected_issues = list(issue_result["issues"][:20])
    comments: dict[int | str, list[dict[str, Any]]] = {}
    for issue in selected_issues:
        number = issue.get("number")
        if number is None:
            issue["eligibility_evidence_complete"] = False
            issue["comments_error"] = "issue number is missing"
            continue
        comment_result = fetch_issue_comments(repo_url, int(number), token=github_token)
        activity_result = fetch_issue_activity(repo_url, int(number), token=github_token)
        comments[number] = list(comment_result.get("comments") or [])
        issue["activity"] = activity_result
        issue["eligibility_evidence_complete"] = bool(
            comment_result.get("ok") and activity_result.get("ok")
        )
        if not comment_result.get("ok"):
            issue["comments_error"] = str(comment_result.get("error") or "comments unavailable")
        if not activity_result.get("ok"):
            issue["activity_error"] = str(activity_result.get("error") or "activity unavailable")
    return selected_issues, comments, None


def _normalize_directions(value: Any, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Discover model top_directions must be a list")
    candidate_numbers = {int(item["number"]) for item in candidates if item.get("number") is not None}
    directions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Discover model directions must be objects")
        source_kind = str(item.get("source_kind") or "")
        issue_number = item.get("issue_number")
        if source_kind == "issue":
            if isinstance(issue_number, bool) or not isinstance(issue_number, int):
                raise ValueError("issue direction must define an integer issue_number")
            if issue_number not in candidate_numbers:
                raise ValueError(f"direction references unknown candidate Issue #{issue_number}")
            direction_id = f"issue:{issue_number}"
        elif source_kind == "architecture":
            if issue_number is not None:
                raise ValueError("architecture direction cannot define issue_number")
            identity = "\n".join(str(item.get(key) or "").strip() for key in ("name", "source", "entry"))
            direction_id = f"architecture:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"
        else:
            raise ValueError("direction source_kind must be issue or architecture")
        direction = ContributionDirection.model_validate({**item, "id": direction_id}).model_dump(mode="json")
        if direction_id in seen_ids:
            raise ValueError(f"duplicate contribution direction id: {direction_id}")
        seen_ids.add(direction_id)
        directions.append(direction)
    return directions
