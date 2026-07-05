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
from osc_agent.tools.repo import (
    analyze_architecture_dimensions,
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
    """创建一次可恢复工作流，并把 run.json 作为后续阶段的唯一恢复入口。"""
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
    """从 run.json 恢复阶段状态，避免用户在 1-4 步之间反复粘贴上下文。"""
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
    agent_review: str | None = None,
) -> ContributionRun:
    """执行第 1 步：读取仓库/Issue/源码证据，生成贡献切入点和可交给 LLM 深挖的提示。"""
    run = create_run(repo_root=repo_root, repo_url=repo_url)
    issues, comments_by_issue, issue_error = _collect_issues(repo_url=repo_url, issues_file=issues_file)
    candidates = filter_candidate_issues(issues, comments_by_issue)
    dimensions = analyze_architecture_dimensions(repo_root=repo_root)
    evidence_pack = build_discover_evidence(repo_root=repo_root)
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
        "evidence_pack": evidence_pack,
        "agent_review": agent_review,
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
    """把 LLM 对第 1 步的深度分析追加进 artifact，用于后续方案设计。"""
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
    agent_design: str | None = None,
) -> ContributionRun:
    """执行第 2 步：基于 discover 产物和选定方向，生成方案，并保存可恢复设计上下文。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    discover = _read_json(run, "01_discover.json")
    selected = direction or run.selected_direction or _default_direction(discover)
    _ensure_direction_is_known(selected, discover)
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
        "options": _design_options(selected, discover),
        "recommended": "方案 1：最小可审查扩展",
        "maintainer_comment": _maintainer_comment(selected),
        "interview_story": _interview_story(selected),
        "agent_design": agent_design,
        "agent_design_prompt": build_design_review_prompt(discover=discover, selected=selected),
    }
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    _write_text(run, "02_design_agent_prompt.md", payload["agent_design_prompt"])
    save_run(run)
    return run


def attach_design_agent_review(*, repo_root: Path, run_id: str, review: str) -> ContributionRun:
    """把 LLM 生成的具体技术方案写回第 2 步 artifact，后续实现和 PR 草稿都会读取它。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    payload = _read_json(run, "02_design.json")
    payload["agent_design"] = review
    _write_json(run, "02_design.json", payload)
    _write_text(run, "02_design.md", render_design(payload))
    return run


def prepare_implementation_stage(*, repo_root: Path, run_id: str) -> tuple[ContributionRun, str]:
    """在调用 agent_loop 前先创建 todo/task 和实现提示，保证第 3 步不是事后补记录。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    run.stage = "implement"
    todo_write(
        [
            {"content": "阅读方案中涉及的核心文件并确认代码风格", "status": "in_progress"},
            {"content": "按推荐方案实现最小可审查改动", "status": "pending"},
            {"content": "运行 focused tests 或手动验证", "status": "pending"},
            {"content": "整理 PR 草稿所需的 Problem/Solution/Testing 信息", "status": "pending"},
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
) -> ContributionRun:
    """实现完成后只更新执行结果，不再创建 todo/task，避免流程顺序被反转。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    design = _read_json(run, "02_design.json")
    existing = _read_text(run, "03_implementation_report.md", default="")
    report = {
        "selected_direction": run.selected_direction,
        "recommended": design.get("recommended"),
        "implementation_prompt": build_implementation_prompt(run, design),
        "created_tasks": [],
        "agent_output": agent_output or "Implementation finished without captured output.",
        "git_status_before": _extract_code_block(existing) or "",
        "git_status_after": git_status(repo_root=repo_root),
        "test_summary": test_summary or _infer_test_summary(agent_output or ""),
    }
    _write_text(run, "03_implementation_report.md", render_implementation_report(report))
    save_run(run)
    return run


def implement_stage(*, repo_root: Path, run_id: str, agent_output: str | None = None) -> ContributionRun:
    """兼容旧调用：无输出时准备实现，有输出时记录实现结果。"""
    if agent_output is None:
        run, _ = prepare_implementation_stage(repo_root=repo_root, run_id=run_id)
        return run
    return record_implementation_result(repo_root=repo_root, run_id=run_id, agent_output=agent_output)


def draft_pr_stage(*, repo_root: Path, run_id: str) -> ContributionRun:
    """执行第 4 步：读取 workflow artifact 和当前 diff，生成只读 PR 草稿。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    run.stage = "draft_pr"
    _write_text(run, "04_pr_draft.md", draft_pr(repo_root=repo_root, run_id=run_id))
    save_run(run)
    return run


def build_discover_evidence(*, repo_root: Path) -> dict[str, Any]:
    """收集源码证据包，让 LLM 深度分析时有具体文件和符号，而不是凭空判断。"""
    return {
        "entrypoints": detect_entrypoints(repo_root=repo_root),
        "planning_symbols": find_functions(repo_root=repo_root, query="plan")[:10],
        "task_symbols": find_functions(repo_root=repo_root, query="task")[:10],
        "tool_symbols": find_functions(repo_root=repo_root, query="tool")[:10],
        "context_symbols": find_functions(repo_root=repo_root, query="context")[:10],
        "trace_symbols": find_functions(repo_root=repo_root, query="trace")[:10],
    }


def build_discover_review_prompt(
    *,
    repo_url: str,
    candidates: list[dict[str, Any]],
    dimensions: list[dict[str, str]],
    evidence_pack: dict[str, Any],
) -> str:
    """生成第 1 步深度分析提示，要求模型按 4 个 md 的标准补足证据和 Top 3。"""
    return (
        "你正在执行 OpenSourcePR 第 1 步：寻找贡献切入点。\n"
        f"GitHub 地址：{repo_url}\n"
        "请基于下面的候选 issue、7 个架构维度和源码证据包，输出严谨分析。\n"
        "要求：每个架构维度必须定位到文件和函数；不能定位时写“未定位到具体实现”；"
        "最后给出 Top 3 贡献建议，说明工作量、风险、面试叙事价值。\n\n"
        f"候选 issue JSON：\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        f"架构维度初筛：\n{json.dumps(dimensions, ensure_ascii=False, indent=2)}\n\n"
        f"源码证据包：\n{json.dumps(evidence_pack, ensure_ascii=False, indent=2)}\n"
    )


def build_design_review_prompt(*, discover: dict[str, Any], selected: str) -> str:
    """生成第 2 步方案设计提示，要求模型重新利用 discover 证据给出具体方案。"""
    return (
        "你正在执行 OpenSourcePR 第 2 步：技术方案设计。\n"
        f"选定贡献方向：{selected}\n"
        "请基于 discover artifact，输出：问题边界、2-3 个方案、方案对比矩阵、推荐方案、"
        "文件级实现计划、验证策略、维护者沟通评论、面试叙事框架。\n"
        "要求方案必须具体到核心文件/函数，单个 PR 尽量控制在 300 行以内。\n\n"
        f"discover artifact：\n{json.dumps(discover, ensure_ascii=False, indent=2)[:20000]}"
    )


def build_implementation_prompt(run: ContributionRun, design: dict[str, Any]) -> str:
    """把第 2 步方案转换成 agent_loop 可执行的实现任务提示。"""
    agent_design = design.get("agent_design") or ""
    return (
        "Follow the OpenSourcePR implementation workflow.\n"
        f"Repository: {run.repo_url}\n"
        f"Selected direction: {run.selected_direction}\n"
        f"Recommended approach: {design.get('recommended')}\n"
        "Before editing, read all files referenced by the design, inspect style/config, and keep changes scoped.\n"
        "Then implement, run focused verification, and report modified files, tests, risks, and PR notes.\n\n"
        f"Detailed design from artifact:\n{agent_design or render_design(design)}"
    )


def implementation_prompt_for_run(*, repo_root: Path, run_id: str) -> str:
    """只读取已保存 artifact 来恢复实现提示，供 CLI 调用 agent_loop 前使用。"""
    run = load_run(repo_root=repo_root, run_id=run_id)
    return build_implementation_prompt(run, _read_json(run, "02_design.json"))


def render_discover(payload: dict[str, Any]) -> str:
    issue_rows = "\n".join(
        f"| #{issue['number']} | {issue['title']} | {', '.join(issue['labels'])} | TBD | 小/中 | 符合筛选条件 |"
        for issue in payload["candidate_issues"]
    ) or "| - | 未找到符合条件的 issue | - | - | - | 建议从架构维度选择 |"
    dimensions = "\n".join(
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
        f"**第 {index} 名：{item['name']}**\n"
        f"- 一句话描述：{item['description']}\n"
        f"- 来源维度：{item['source']}\n"
        f"- 入口文件：{item['entry']}\n"
        "- 为什么适合我：匹配 Python/TypeScript/Agent 工程分析能力。\n"
        f"- 预计工作量：{item['effort']}\n"
        f"- 面试中能讲什么：{item['interview']}\n"
        "- 风险点：需要维护者确认范围。"
        for index, item in enumerate(payload["top_directions"], start=1)
    )
    review = f"\n## Agent 深度分析\n\n{payload['agent_review']}\n" if payload.get("agent_review") else ""
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
        f"{dimensions}\n"
        f"{review}\n"
        "## Top 3 贡献建议\n\n"
        f"{directions}\n"
    )


def render_design(payload: dict[str, Any]) -> str:
    options = "\n\n".join(
        f"### {option['name']}\n**核心思路：** {option['idea']}\n**优点：** {option['pros']}\n**缺点 / 风险：** {option['cons']}"
        for option in payload["options"]
    )
    agent_design = f"\n## Agent 具体方案\n\n{payload['agent_design']}\n" if payload.get("agent_design") else ""
    return (
        "# 技术方案设计\n\n"
        "## 问题边界定义\n"
        f"**要解决的核心问题：** {payload['problem_boundary']}\n"
        f"**不在本次 PR 范围内的问题：** {'；'.join(payload['out_of_scope'])}\n"
        f"**成功标准：** {'；'.join(payload['success_criteria'])}\n\n"
        "## 方案设计\n"
        f"{options}\n\n"
        f"**推荐方案：** {payload['recommended']}\n"
        f"{agent_design}\n"
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
        "## Testing\n"
        f"{report['test_summary']}\n\n"
        "## Git 状态（执行前）\n"
        f"```text\n{report.get('git_status_before', '')}\n```\n\n"
        "## Git 状态（执行后）\n"
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


def _ensure_direction_is_known(selected: str, discover: dict[str, Any]) -> None:
    known = {item["name"] for item in discover.get("top_directions", [])}
    if known and selected not in known:
        # 允许用户自定义方向，但把偏离 Top 3 明确留在 artifact 的输入中，不静默替换。
        return


def _design_options(selected: str, discover: dict[str, Any]) -> list[dict[str, str]]:
    entry = next((item["entry"] for item in discover.get("top_directions", []) if item["name"] == selected), "核心文件待确认")
    return [
        {
            "name": "方案 1：最小可审查扩展",
            "idea": f"围绕“{selected}”从 {entry} 附近切入，优先复用现有抽象，只新增必要接口和测试。",
            "pros": "改动小，容易 review，适合 first PR。",
            "cons": "覆盖面有限，需要后续 PR 继续完善。",
        },
        {
            "name": "方案 2：策略化增强",
            "idea": "把相关行为抽成策略或 helper，降低后续扩展成本。",
            "pros": "扩展性更好。",
            "cons": "实现复杂度更高，可能超过单个 PR 范围。",
        },
        {
            "name": "方案 3：测试和文档先行",
            "idea": "先补最小复现、测试或文档，为后续功能 PR 降低维护成本。",
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
        f"我先从 issue 和架构维度定位到“{selected}”，再比较最小扩展、策略化和测试文档优先三种方案，"
        "最终选择最小可审查方案来体现范围控制、接口设计和验证驱动实现能力。"
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
    path = Path(run.artifacts_dir, name)
    return path.read_text(encoding="utf-8") if path.exists() else default


def _write_text(run: ContributionRun, name: str, value: str) -> None:
    path = Path(run.artifacts_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
