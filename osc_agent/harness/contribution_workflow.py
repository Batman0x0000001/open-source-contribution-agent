"""
开始
↓
discover_stage()
分析仓库
分析 Issue
生成贡献方向
↓
design_stage()
确定贡献方向
生成技术方案
↓
implement_stage()
生成实现 Prompt
创建 Todo
交给 Agent 修改代码
↓
draft_pr_stage()
读取 Git Diff
生成 PR Draft
↓
结束
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import secrets
from pathlib import Path
from typing import Any

from osc_agent.harness.tasks import create_default_task_graph
from osc_agent.harness.todo import todo_write
from osc_agent.tools.github import (
    CANDIDATE_LABELS,
    fetch_issue_comments,
    fetch_issues,
    filter_candidate_issues,
    load_issues_file,
)
from osc_agent.tools.git import git_diff, git_status
from osc_agent.tools.pr import draft_pr
from osc_agent.tools.repo import analyze_architecture_dimensions, detect_entrypoints, inspect_repo, repo_tree

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
    """创建一次可恢复的贡献工作流运行，并把 run.json 作为恢复入口落盘。"""
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"
    artifacts_dir = _runs_dir(repo_root) / run_id
    run = ContributionRun(
        run_id=run_id,
        repo_root=str(repo_root.resolve()),
        repo_url=repo_url,
        stage="discover",
        selected_direction=None,
        artifacts_dir=str(artifacts_dir),
    )
    save_run(run)
    return run


def load_run(*, repo_root: Path, run_id: str) -> ContributionRun:
    """从 run.json 恢复工作流状态，后续阶段不需要用户重新粘贴前序结果。"""
    path = _runs_dir(repo_root) / run_id / "run.json"
    if not path.exists():
        raise ValueError(f"contribution run not found: {run_id}")
    return ContributionRun(**json.loads(path.read_text(encoding="utf-8")))


def save_run(run: ContributionRun) -> None:
    artifacts = Path(run.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "run.json").write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_stage(
    *,
    repo_root: Path,
    repo_url: str,
    issues_file: Path | None = None,
) -> ContributionRun:
    """执行第 1 步：读取仓库和 issue，生成候选贡献方向并保存 discover 产物。"""
    run = create_run(repo_root=repo_root, repo_url=repo_url)
    issues, comments_by_issue, issue_error = _collect_issues(repo_url=repo_url, issues_file=issues_file)
    candidates = filter_candidate_issues(issues, comments_by_issue)
    dimensions = analyze_architecture_dimensions(repo_root=repo_root)
    directions = _top_directions(candidates, dimensions)
    payload = {
        "repo_url": repo_url,
        "repo_overview": inspect_repo(repo_root=repo_root),
        "tree": repo_tree(repo_root=repo_root, depth=3),
        "entrypoints": detect_entrypoints(repo_root=repo_root),
        "candidate_issues": candidates,
        "architecture_dimensions": dimensions,
        "top_directions": directions,
        "issue_error": issue_error,
    }
    _write_json(run, "01_discover.json", payload)
    _write_text(run, "01_discover.md", render_discover(payload))
    save_run(run)
    return run


def design_stage(*, repo_root: Path, run_id: str, direction: str | None = None) -> ContributionRun:
    """执行第 2 步：读取 discover 产物和选定方向，生成技术方案设计。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    discover = _read_json(run, "01_discover.json")
    selected = direction or run.selected_direction or _default_direction(discover)
    run.selected_direction = selected
    run.stage = "design"
    payload = {
        "selected_direction": selected,
        "problem_boundary": f"围绕“{selected}”完成一个小而可审查的开源贡献。",
        "out_of_scope": [
            "不自动 push、commit 或 open PR。",
            "不引入真实 GitHub 写操作。",
            "不做跨模块大规模重构。",
        ],
        "success_criteria": [
            "改动范围控制在 1-3 个核心文件附近。",
            "有 focused tests 或明确手动验证步骤。",
            "PR 草稿能说明 Problem、Solution、Testing 和 reviewer 关注点。",
        ],
        "options": _design_options(selected),
        "recommended": "方案 1：最小可审查扩展",
        "maintainer_comment": _maintainer_comment(selected),
        "interview_story": _interview_story(selected),
    }
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    save_run(run)
    return run


def implement_stage(*, repo_root: Path, run_id: str, agent_output: str | None = None) -> ContributionRun:
    """执行第 3 步：创建任务模板和实现提示，实际代码修改由现有 agent_loop 完成。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    run.stage = "implement"
    todo_write(
        [
            {"content": "Read files named by the design artifact", "status": "completed"},
            {"content": "Implement the recommended scoped change", "status": "in_progress"},
            {"content": "Run focused verification", "status": "pending"},
            {"content": "Draft PR notes", "status": "pending"},
        ],
        repo_root=repo_root,
    )
    created_tasks = create_default_task_graph(repo_root)
    report = {
        "selected_direction": run.selected_direction,
        "recommended": design.get("recommended"),
        "implementation_prompt": build_implementation_prompt(run, design),
        "created_tasks": [asdict(task) for task in created_tasks],
        "agent_output": agent_output or "Implementation prompt prepared; execute with agent_loop to make code changes.",
        "git_status": git_status(repo_root=repo_root),
    }
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)
    return run


def draft_pr_stage(*, repo_root: Path, run_id: str) -> ContributionRun:
    """执行第 4 步：读取 workflow artifact 和 diff，生成只读 PR 草稿。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    run.stage = "draft_pr"
    pr_body = draft_pr(repo_root=repo_root, run_id=run_id)
    _write_text(run, "04_pr_draft.md", pr_body)
    save_run(run)
    return run


def build_implementation_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    """把第 2 步方案转换成 agent_loop 可执行的实现任务提示。"""
    return (
        "Follow the OpenSourcePR implementation workflow.\n"
        f"Repository: {run.repo_url}\n"
        f"Selected direction: {run.selected_direction}\n"
        f"Recommended approach: {design.get('recommended')}\n"
        "First read the relevant files, preserve current public APIs, implement the smallest scoped change, "
        "run focused tests, then summarize modified files, tests, risks, and PR notes."
    )


def implementation_prompt_for_run(*, repo_root: Path, run_id: str) -> str:
    """只读取已保存 artifact 来恢复实现提示，供 CLI 调用 agent_loop 前使用。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    return build_implementation_prompt(run, design)


def render_discover(payload: dict[str, Any]) -> str:
    issue_rows = "\n".join(
        f"| #{issue['number']} | {issue['title']} | {', '.join(issue['labels'])} | TBD | 小/中 | 符合筛选条件 |"
        for issue in payload["candidate_issues"]
    ) or "| - | 未找到符合条件的 issue | - | - | - | 建议从架构维度选择 |"
    dimension_sections = "\n".join(
        "\n".join(
            [
                f"### {item['dimension']}",
                f"**现状描述：** {item['current']}",
                f"**缺陷 / 缺失：** {item['gap']}",
                f"**影响程度：** {item['impact']}",
                f"**改进方向：** {item['improvement']}",
                f"**改动范围：** {item['scope']}",
                f"**面试叙事角度：** {item['interview_angle']}",
            ]
        )
        for item in payload["architecture_dimensions"]
    )
    directions = "\n".join(
        f"**第 {index} 名：{item['name']}**\n- 一句话描述：{item['description']}\n- 来源维度：{item['source']}\n- 入口文件：{item['entry']}\n- 为什么适合我：匹配 Python/TypeScript/Agent 工程分析能力。\n- 预计工作量：{item['effort']}\n- 面试中能讲什么：{item['interview']}\n- 风险点：需要维护者确认范围。"
        for index, item in enumerate(payload["top_directions"], start=1)
    )
    return (
        "# 开源项目贡献分析\n\n"
        f"## 项目信息\nGitHub 地址：{payload['repo_url']}\n\n"
        "## 准备工作\n"
        f"```text\n{payload['repo_overview']}\n\n{payload['tree']}\n```\n\n"
        f"入口文件：{', '.join(payload['entrypoints']) or '未定位到具体实现'}\n\n"
        "## Issue 列表筛选\n\n"
        "| Issue # | 标题 | 类型 | 所需技能 | 预估工作量 | 推荐理由 |\n"
        "|---|---|---|---|---|---|\n"
        f"{issue_rows}\n\n"
        "## 架构层缺陷分析\n\n"
        f"{dimension_sections}\n\n"
        "## Top 3 贡献建议\n\n"
        f"{directions}\n"
    )


def render_design(payload: dict[str, Any]) -> str:
    options = "\n\n".join(
        f"### {option['name']}\n**核心思路：** {option['idea']}\n**优点：** {option['pros']}\n**缺点 / 风险：** {option['cons']}"
        for option in payload["options"]
    )
    return (
        "# 技术方案设计\n\n"
        "## 问题边界定义\n"
        f"**要解决的核心问题：** {payload['problem_boundary']}\n"
        f"**不在本次 PR 范围内的问题：** {'；'.join(payload['out_of_scope'])}\n"
        f"**成功标准：** {'；'.join(payload['success_criteria'])}\n\n"
        "## 方案设计\n"
        f"{options}\n\n"
        f"**推荐方案：** {payload['recommended']}\n\n"
        "## 与维护者沟通前的准备\n"
        f"{payload['maintainer_comment']}\n\n"
        "## 面试叙事框架\n"
        f"{payload['interview_story']}\n"
    )


def render_implementation_report(report: dict[str, Any]) -> str:
    return (
        "# 技术方案实现\n\n"
        f"## 选定方向\n{report['selected_direction']}\n\n"
        f"## 推荐方案\n{report['recommended']}\n\n"
        "## 实现提示\n"
        f"```text\n{report['implementation_prompt']}\n```\n\n"
        "## Agent 输出\n"
        f"{report['agent_output']}\n\n"
        "## Git 状态\n"
        f"```text\n{report['git_status']}\n```\n"
    )


def _collect_issues(*, repo_url: str, issues_file: Path | None) -> tuple[list[dict[str, Any]], dict[int | str, list[dict[str, Any]]], str | None]:
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
                "description": "围绕已有 issue 做小范围修复或增强。",
                "source": f"Issue #{issue['number']}",
                "entry": issue.get("url") or "issue",
                "effort": "小/中",
                "interview": "体现需求澄清、范围控制和测试验证能力。",
            }
        )
    for item in dimensions:
        if len(directions) >= 3:
            break
        directions.append(
            {
                "name": f"改进{item['dimension']}",
                "description": item["improvement"],
                "source": f"代码分析-{item['dimension']}",
                "entry": item["location"],
                "effort": "中",
                "interview": item["interview_angle"],
            }
        )
    return directions[:3]


def _default_direction(discover: dict[str, Any]) -> str:
    directions = discover.get("top_directions") or []
    if not directions:
        raise ValueError("discover artifact has no contribution directions")
    return directions[0]["name"]


def _design_options(selected: str) -> list[dict[str, str]]:
    return [
        {
            "name": "方案 1：最小可审查扩展",
            "idea": f"围绕“{selected}”新增最小接口或工具能力，优先复用现有 harness。",
            "pros": "改动小，容易 review，适合 first PR。",
            "cons": "覆盖面有限，需要后续 PR 继续完善。",
        },
        {
            "name": "方案 2：配置化增强",
            "idea": "把能力做成可配置策略，减少硬编码。",
            "pros": "扩展性更好。",
            "cons": "实现复杂度更高，可能超过单个 PR 范围。",
        },
        {
            "name": "方案 3：文档和测试优先",
            "idea": "先补文档、测试或示例，为后续功能 PR 降低维护成本。",
            "pros": "风险低，适合与维护者建立信任。",
            "cons": "技术深度相对有限。",
        },
    ]


def _maintainer_comment(selected: str) -> str:
    return (
        f"I noticed a scoped opportunity around {selected}. I am considering a small PR that preserves the "
        "current public API, adds focused tests, and documents the behavior. Would this direction be useful "
        "for the project, or is there an existing plan I should align with?"
    )


def _interview_story(selected: str) -> str:
    return (
        f"我先从 issue 和架构维度定位到“{selected}”，再比较最小扩展、配置化和文档测试优先三种方案，"
        "最终选择最小可审查方案来体现范围控制、接口设计和验证驱动实现能力。"
    )


def _runs_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "contribution_runs"


def _write_json(run: ContributionRun, name: str, value: dict[str, Any]) -> None:
    Path(run.artifacts_dir, name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(run: ContributionRun, name: str) -> dict[str, Any]:
    path = Path(run.artifacts_dir, name)
    if not path.exists():
        raise ValueError(f"required artifact missing: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(run: ContributionRun, name: str, value: str) -> None:
    Path(run.artifacts_dir, name).write_text(value.rstrip() + "\n", encoding="utf-8")
