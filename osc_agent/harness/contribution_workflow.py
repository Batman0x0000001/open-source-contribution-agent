from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import secrets
from pathlib import Path
from typing import Any, Callable

from osc_agent.harness.tasks import create_default_task_graph
from osc_agent.harness.todo import todo_write
from osc_agent.tools.github import (
    CANDIDATE_LABELS,
    apply_issue_scores,
    fetch_issue_comments,
    fetch_issues,
    filter_candidate_issues,
    load_issues_file,
)
from osc_agent.tools.git import git_status
from osc_agent.tools.pr import draft_pr
from osc_agent.tools.repo import (
    analyze_architecture_dimensions,
    collect_repo_evidence_pack,
    detect_entrypoints,
    find_functions,
    inspect_repo,
    repo_tree,
)

STAGES = {"discover", "design", "implement", "draft_pr"}


@dataclass
class ContributionRun:
    run_id: str
    repo_root: str
    repo_url: str
    stage: str
    selected_direction: str | None
    artifacts_dir: str


def create_run(*, repo_root: Path, repo_url: str) -> ContributionRun:
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"
    run = ContributionRun(
        run_id=run_id,
        repo_root=str(repo_root.resolve()),
        repo_url=repo_url,
        stage="discover",
        selected_direction=None,
        artifacts_dir=str(_runs_dir(repo_root) / run_id),
    )
    save_run(run)
    return run


def load_run(*, repo_root: Path, run_id: str) -> ContributionRun:
    path = _runs_dir(repo_root) / run_id / "run.json"
    if not path.exists():
        raise ValueError(f"contribution run not found: {run_id}")
    return ContributionRun(**json.loads(path.read_text(encoding="utf-8")))


def save_run(run: ContributionRun) -> None:
    artifacts = Path(run.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    _write_raw_json(artifacts / "run.json", asdict(run))


def discover_stage(
    *,
    repo_root: Path,
    repo_url: str,
    issues_file: Path | None = None,
    client: Any | None = None,
    settings: Any | None = None,
    agent_review: str | None = None,
) -> ContributionRun:
    run = create_run(repo_root=repo_root, repo_url=repo_url)
    issues, comments_by_issue, issue_error = _collect_issues(repo_url=repo_url, issues_file=issues_file)
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
        "evidence_pack": evidence_pack,
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
    save_run(run)
    return run


def attach_discover_agent_review(*, repo_root: Path, run_id: str, review: str) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    payload = _read_json(run, "01_discover.json")
    payload["agent_review"] = review
    _write_json(run, "01_discover.json", payload)
    _write_text(run, "01_discover.md", render_discover(payload))
    return run


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
    discover = _read_json(run, "01_discover.json")
    selected = direction or run.selected_direction or _default_direction(discover)
    _ensure_direction_is_known(selected, discover)
    run.selected_direction = selected
    run.stage = "design"

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
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    _write_text(run, "02_design_agent_prompt.md", payload["agent_design_prompt"])
    save_run(run)
    return run


def attach_design_agent_review(*, repo_root: Path, run_id: str, review: str) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    payload = _read_json(run, "02_design.json")
    payload["agent_design"] = review
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    return run


def prepare_implementation_stage(*, repo_root: Path, run_id: str) -> tuple[ContributionRun, str]:
    run = load_run(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    run.stage = "implement"
    todo_write(
        [
            {"content": "Read the selected design and confirm implementation scope", "status": "in_progress"},
            {"content": "Implement the smallest reviewable change", "status": "pending"},
            {"content": "Run focused tests or document manual verification", "status": "pending"},
            {"content": "Summarize files, tests, risks, and PR notes", "status": "pending"},
        ],
        repo_root=repo_root,
    )
    tasks = create_default_task_graph(repo_root)
    prompt = build_implementation_prompt(run, design)
    report = {
        "selected_direction": run.selected_direction,
        "recommended": design.get("recommended"),
        "implementation_prompt": prompt,
        "created_tasks": [asdict(task) for task in tasks],
        "agent_output": "Implementation has not run yet.",
        "git_status_before": git_status(repo_root=repo_root),
        "git_status_after": "",
        "test_summary": "Not run yet.",
    }
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)
    return run, prompt


def record_implementation_result(
    *,
    repo_root: Path,
    run_id: str,
    agent_output: str | None = None,
    test_summary: str | None = None,
    understanding_output: str | None = None,
    verification_output: str | None = None,
) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    existing = _read_text(run, "03_implementation_report.md", default="")
    report = {
        "selected_direction": run.selected_direction,
        "recommended": design.get("recommended"),
        "implementation_prompt": build_implementation_prompt(run, design),
        "created_tasks": [],
        "understanding_output": understanding_output or "",
        "agent_output": agent_output or "Implementation finished without captured output.",
        "verification_output": verification_output or "",
        "git_status_before": _extract_code_block(existing) or "",
        "git_status_after": git_status(repo_root=repo_root),
        "test_summary": test_summary or _infer_test_summary("\n".join([agent_output or "", verification_output or ""])),
    }
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)
    return run


def execute_implementation_stage(
    *,
    repo_root: Path,
    run_id: str,
    run_step: Callable[[str, str], str],
) -> ContributionRun:
    """Run the implementation substeps in order with an explicit edit checkpoint."""
    run, fallback_prompt = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")

    understanding = run_step("understanding", build_understanding_prompt(run, design))
    if "READY_TO_EDIT" not in understanding:
        raise ValueError(
            "Implementation stopped at the understanding checkpoint: "
            "the agent did not confirm READY_TO_EDIT."
        )

    edit_prompt = build_edit_prompt(run, design, understanding) or fallback_prompt
    agent_output = run_step("edit", edit_prompt)
    verification = run_step("verification", build_verification_prompt(run, design))
    return record_implementation_result(
        repo_root=repo_root,
        run_id=run_id,
        understanding_output=understanding,
        agent_output=agent_output,
        verification_output=verification,
    )


def implement_stage(*, repo_root: Path, run_id: str, agent_output: str | None = None) -> ContributionRun:
    if agent_output is None:
        run, _ = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
        return run
    return record_implementation_result(repo_root=repo_root, run_id=run_id, agent_output=agent_output)


def draft_pr_stage(
    *,
    repo_root: Path,
    run_id: str,
    client: Any | None = None,
    settings: Any | None = None,
) -> ContributionRun:
    run = load_run(repo_root=repo_root, run_id=run_id)
    run.stage = "draft_pr"
    _write_text(run, "04_pr_draft.md", draft_pr(repo_root=repo_root, run_id=run_id, client=client, settings=settings))
    save_run(run)
    return run


def build_discover_evidence(*, repo_root: Path) -> dict[str, Any]:
    return collect_repo_evidence_pack(repo_root=repo_root)


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


def build_design_review_prompt(*, discover: dict[str, Any], selected: str) -> str:
    focused = _focused_discover_for_design(discover, selected)
    return (
        "OpenSourcePR step 2: produce a concrete technical design.\n"
        f"Selected direction: {selected}\n\n"
        "Only use evidence relevant to the selected direction. Ignore other candidate issues unless they "
        "directly explain the selected one.\n\n"
        f"Focused discover evidence:\n{json.dumps(focused, ensure_ascii=False, indent=2)[:12000]}"
    )


def build_understanding_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    return (
        "OpenSourcePR implementation step 3a: understand the task before editing.\n"
        "Do not modify files in this step.\n"
        f"Selected direction: {run.selected_direction}\n"
        f"Files to inspect: {', '.join(design.get('files_to_modify') or ['not specified'])}\n"
        "Read the referenced files, summarize the implementation boundary, and explicitly say READY_TO_EDIT "
        "only if the plan is concrete enough."
    )


def build_edit_prompt(run: ContributionRun, design: dict[str, Any], understanding: str) -> str:
    return (
        "OpenSourcePR implementation step 3b: edit the code.\n"
        "Before editing, verify the referenced files and local style one more time.\n"
        "Keep changes within the approved scope unless the repository proves the design inaccurate.\n"
        f"Repository: {run.repo_url}\n"
        f"Selected direction: {run.selected_direction}\n"
        f"Recommended approach: {design.get('recommended')}\n\n"
        f"Understanding checkpoint:\n{understanding}\n\n"
        f"Detailed design:\n{design.get('agent_design') or render_design(design)}"
    )


def build_verification_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    tests = design.get("tests_to_run") or ["run the narrowest relevant pytest command or document why none applies"]
    return (
        "OpenSourcePR implementation step 3c: verify the change.\n"
        "Run focused verification, inspect git diff/status, and report exact commands and results.\n"
        f"Expected tests: {json.dumps(tests, ensure_ascii=False)}\n"
        "Do not open a PR, push, or commit."
    )


def build_implementation_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    return build_edit_prompt(run, design, understanding="Prepared from saved workflow artifacts.")


def implementation_prompt_for_run(*, repo_root: Path, run_id: str) -> str:
    run = load_run(repo_root=repo_root, run_id=run_id)
    return build_implementation_prompt(run, _read_json(run, "02_design.json"))


def validate_design_files(repo_root: Path, design: dict[str, Any]) -> dict[str, Any]:
    files = _extract_design_files(design)
    missing = [path for path in files if not (repo_root / path).exists()]
    return {
        "ok": not missing,
        "files": files,
        "missing_files": missing,
    }


def render_discover(payload: dict[str, Any]) -> str:
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
        "| Issue | Title | Labels |\n|---|---|---|\n"
        f"{issue_rows}\n\n"
        "## Architecture Gap Analysis\n"
        f"{dimensions}\n"
        f"{review}\n"
        "## Top 3 Contribution Suggestions\n\n"
        f"{directions}\n"
    )


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


def render_implementation_report(report: dict[str, Any]) -> str:
    return (
        "# Implementation Report\n\n"
        f"## Selected Direction\n{report['selected_direction']}\n\n"
        f"## Recommended Approach\n{report['recommended']}\n\n"
        "## Implementation Prompt\n"
        f"```text\n{report['implementation_prompt']}\n```\n\n"
        "## Understanding\n"
        f"{report.get('understanding_output', '')}\n\n"
        "## Agent Output\n"
        f"{report['agent_output']}\n\n"
        "## Verification\n"
        f"{report.get('verification_output', '')}\n\n"
        "## Testing\n"
        f"{report['test_summary']}\n\n"
        "## Git Status Before\n"
        f"```text\n{report.get('git_status_before', '')}\n```\n\n"
        "## Git Status After\n"
        f"```text\n{report.get('git_status_after', '')}\n```\n"
    )


def _collect_issues(
    *,
    repo_url: str,
    issues_file: Path | None,
) -> tuple[list[dict[str, Any]], dict[int | str, list[dict[str, Any]]], str | None]:
    if issues_file is not None:
        issues, comments = load_issues_file(str(issues_file))
        return issues, comments, None
    issue_result = fetch_issues(repo_url, labels=sorted(CANDIDATE_LABELS))
    if not issue_result["ok"]:
        return [], {}, issue_result["error"]
    comments: dict[int | str, list[dict[str, Any]]] = {}
    for issue in issue_result["issues"][:20]:
        number = issue.get("number")
        if number is None:
            continue
        result = fetch_issue_comments(repo_url, int(number))
        comments[number] = result.get("comments", [])
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
        }
    else:
        payload = template
    payload["validation"] = validate_design_files(repo_root, payload)
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


def _normalize_directions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and item.get("name") and item.get("description")]


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


def _infer_test_summary(agent_output: str) -> str:
    lowered = agent_output.lower()
    if "pytest" in lowered or "passed" in lowered or "failed" in lowered:
        return agent_output[-2000:]
    return "No explicit test command found in captured agent output."


def _extract_code_block(text: str) -> str:
    marker = "```text"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = text.find("```", start)
    return text[start:end].strip() if end != -1 else ""


def _try_import_stage_agents():
    try:
        from osc_agent.harness import stage_agents

        return stage_agents
    except ImportError:
        return None


def _runs_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "contribution_runs"


def _write_json(run: ContributionRun, name: str, value: dict[str, Any]) -> None:
    _write_raw_json(Path(run.artifacts_dir) / name, value)


def _write_raw_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(run: ContributionRun, name: str) -> dict[str, Any]:
    path = Path(run.artifacts_dir, name)
    if not path.exists():
        raise ValueError(f"required artifact missing: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(run: ContributionRun, name: str, default: str = "") -> str:
    path = Path(run.artifacts_dir) / name
    return path.read_text(encoding="utf-8") if path.exists() else default


def _write_text(run: ContributionRun, name: str, value: str) -> None:
    path = Path(run.artifacts_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
