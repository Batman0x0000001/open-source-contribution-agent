"""
收集阶段证据
    ↓
构造专用 prompt + tool schema
    ↓
调用 Anthropic tool_use
    ↓
提取结构化 JSON 输出
    ↓
返回给 workflow 落盘
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osc_agent.config import Settings
from osc_agent.harness.trace import append_trace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    """兼容 Anthropic SDK 对象和测试里的 dict block。"""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _call_with_tool(
    client: Any,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    tool_def: dict[str, Any],
    tool_name: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """向 Anthropic 发送单轮 tool_use 请求并提取结构化输出。

    Returns the ``input`` dict from the first matching tool_use block,
    or ``None`` if the model did not return one.
    """
    try:
        response = client.messages.create(
            model=settings.model_id,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool_def],
            max_tokens=8000,
        )
    except Exception as exc:  # noqa: BLE001
        if repo_root is not None:
            append_trace(repo_root, "stage_agent_error", {"error": str(exc)})
        return None

    for block in response.content:
        if _block_attr(block, "type") == "tool_use" and _block_attr(block, "name") == tool_name:
            result = _block_attr(block, "input", {})
            if repo_root is not None:
                append_trace(
                    repo_root,
                    f"stage_agent_{tool_name}",
                    {"keys": list(result.keys()) if isinstance(result, dict) else []},
                )
            return result if isinstance(result, dict) else None

    if repo_root is not None:
        append_trace(repo_root, "stage_agent_no_tool_use", {"expected": tool_name})
    return None


# ---------------------------------------------------------------------------
# Tool schema definitions
# ---------------------------------------------------------------------------

_SUBMIT_ANALYSIS_TOOL: dict[str, Any] = {
    "name": "submit_analysis",
    "description": "Submit the structured analysis of contribution directions for the repository.",
    "input_schema": {
        "type": "object",
        "properties": {
            "top_directions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "source": {"type": "string"},
                        "entry": {"type": "string"},
                        "effort": {"type": "string"},
                        "interview": {"type": "string"},
                        "risk": {"type": "string"},
                    },
                    "required": ["name", "description", "source", "entry", "effort", "interview", "risk"],
                },
            },
            "analysis_summary": {"type": "string"},
            "architecture_insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "dimension": {"type": "string"},
                        "current": {"type": "string"},
                        "gap": {"type": "string"},
                        "impact": {"type": "string"},
                        "improvement": {"type": "string"},
                        "scope": {"type": "string"},
                        "interview_angle": {"type": "string"},
                        "location": {"type": "string"},
                    },
                    "required": [
                        "dimension", "current", "gap", "impact",
                        "improvement", "scope", "interview_angle", "location",
                    ],
                },
            },
        },
        "required": ["top_directions", "analysis_summary", "architecture_insights"],
        "additionalProperties": False,
    },
}

_SUBMIT_DESIGN_TOOL: dict[str, Any] = {
    "name": "submit_design",
    "description": "Submit the technical design and implementation plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "problem_boundary": {"type": "string"},
            "out_of_scope": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "idea": {"type": "string"},
                        "pros": {"type": "string"},
                        "cons": {"type": "string"},
                    },
                    "required": ["name", "idea", "pros", "cons"],
                },
            },
            "recommended": {"type": "string"},
            "implementation_plan": {"type": "string"},
            "files_to_modify": {"type": "array", "items": {"type": "string"}},
            "tests_to_run": {"type": "array", "items": {"type": "string"}},
            "maintainer_comment": {"type": "string"},
            "interview_story": {"type": "string"},
        },
        "required": [
            "problem_boundary", "out_of_scope", "success_criteria",
            "options", "recommended", "implementation_plan", "files_to_modify", "tests_to_run",
            "maintainer_comment", "interview_story",
        ],
        "additionalProperties": False,
    },
}

_SUBMIT_ISSUE_SCORES_TOOL: dict[str, Any] = {
    "name": "submit_issue_scores",
    "description": "Submit issue-level contribution feasibility scores.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "title": {"type": "string"},
                        "score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "feasible": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "risk": {"type": "string"},
                    },
                    "required": ["number", "title", "score", "feasible", "reason", "risk"],
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}

_SUBMIT_PR_DRAFT_TOOL: dict[str, Any] = {
    "name": "submit_pr_draft",
    "description": "Submit a structured PR description ready for opening on GitHub.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "problem": {"type": "string"},
            "solution": {"type": "string"},
            "changes": {"type": "array", "items": {"type": "string"}},
            "testing": {"type": "string"},
            "reviewer_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "problem", "solution", "changes", "testing", "reviewer_notes"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Stage 1 — Discover
# ---------------------------------------------------------------------------

_DISCOVER_SYSTEM = (
    "你是一位资深开源贡献顾问。根据提供的仓库概览、目录结构、入口点、候选 Issue、"
    "架构维度分析以及符号级证据，分析该仓库最有价值的贡献方向。\n"
    "要求：\n"
    "1. 从代码质量、功能缺失、文档完善、测试覆盖、性能优化等角度给出 3-7 个具体方向。\n"
    "2. 对每个方向说明来源（issue/代码/文档）、入口文件、预估工作量、面试故事价值和风险。\n"
    "3. 同时输出架构层面的洞察，指出当前状态、差距、改进建议及其面试讲述角度。\n"
    "4. 使用 submit_analysis 工具以结构化 JSON 输出。"
)

_ISSUE_SCORE_SYSTEM = (
    "You are screening GitHub issues for a first open-source contribution. "
    "Read each issue body and comments, then score whether it is clear, unclaimed, scoped, "
    "testable, and likely reviewable in a small PR. Use submit_issue_scores."
)


def _format_discover_prompt(evidence_pack: dict[str, Any]) -> str:
    """将证据包格式化为 user prompt 文本。"""
    sections: list[str] = []

    for key in ("repo_overview", "tree", "entrypoints", "candidate_issues",
                "architecture_dimensions"):
        value = evidence_pack.get(key)
        if value is not None:
            sections.append(f"## {key}\n{value}")

    symbols = evidence_pack.get("evidence_pack", {}).get("symbols")
    if symbols:
        sections.append(f"## symbols\n{json.dumps(symbols, ensure_ascii=False, default=str)[:12000]}")

    return "\n\n".join(sections) if sections else json.dumps(evidence_pack, ensure_ascii=False, default=str)


def run_discover_analysis(
    client: Any,
    settings: Settings,
    evidence_pack: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """调用 LLM 分析贡献方向，返回结构化分析结果或 None。"""
    user_prompt = _format_discover_prompt(evidence_pack)
    return _call_with_tool(
        client,
        settings,
        _DISCOVER_SYSTEM,
        user_prompt,
        _SUBMIT_ANALYSIS_TOOL,
        "submit_analysis",
        repo_root=repo_root,
    )


def score_candidate_issues(
    client: Any,
    settings: Settings,
    candidates: list[dict[str, Any]],
    comments_by_issue: dict[int | str, list[dict[str, Any]]],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """让 LLM 对候选 issue 做二次可行性评分；失败时返回空列表。"""
    if not candidates:
        return []
    issues = []
    for issue in candidates[:20]:
        number = issue.get("number")
        issues.append(
            {
                "number": number,
                "title": issue.get("title", ""),
                "labels": issue.get("labels", []),
                "body": issue.get("body", ""),
                "comments": comments_by_issue.get(number, comments_by_issue.get(str(number), [])),
            }
        )
    result = _call_with_tool(
        client,
        settings,
        _ISSUE_SCORE_SYSTEM,
        json.dumps({"issues": issues}, ensure_ascii=False, indent=2, default=str)[:30000],
        _SUBMIT_ISSUE_SCORES_TOOL,
        "submit_issue_scores",
        repo_root=repo_root,
    )
    scores = result.get("scores") if result else None
    return scores if isinstance(scores, list) else []


# ---------------------------------------------------------------------------
# Stage 2 — Design
# ---------------------------------------------------------------------------

_DESIGN_SYSTEM = (
    "你是一位高级软件架构师。根据前一阶段的贡献方向分析和用户选定的方向，"
    "生成一份具体的技术设计和实施方案。\n"
    "要求：\n"
    "1. 明确问题边界和不在范围内的事项。\n"
    "2. 列出成功标准。\n"
    "3. 给出 2-3 个实现方案并比较优劣，推荐最佳方案。\n"
    "4. 编写文件级别的详细实施计划。\n"
    "5. 撰写一段可直接发布到 GitHub Issue 的英文评论。\n"
    "6. 撰写一段中文面试叙事，用于展示技术决策过程。\n"
    "7. 使用 submit_design 工具以结构化 JSON 输出。"
)


def run_design_generation(
    client: Any,
    settings: Settings,
    discover_payload: dict[str, Any],
    selected_direction: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """调用 LLM 生成技术设计方案，返回结构化设计或 None。"""
    context_str = json.dumps(discover_payload, ensure_ascii=False, default=str)[:20000]
    user_prompt = (
        f"## 前序分析结果（截断）\n{context_str}\n\n"
        f"## 用户选定方向\n{selected_direction}\n\n"
        "请基于以上信息生成详细技术设计。"
    )
    return _call_with_tool(
        client,
        settings,
        _DESIGN_SYSTEM,
        user_prompt,
        _SUBMIT_DESIGN_TOOL,
        "submit_design",
        repo_root=repo_root,
    )


# ---------------------------------------------------------------------------
# Stage 4 — PR Draft
# ---------------------------------------------------------------------------

_PR_DRAFT_SYSTEM = (
    "You are an expert open source contributor writing a pull request description. "
    "Given the selected contribution direction, design summary, implementation report, "
    "git diff, and changed file list, produce a clear, professional PR description.\n"
    "Requirements:\n"
    "1. Title should be concise and follow conventional commit style.\n"
    "2. Clearly state the problem being solved.\n"
    "3. Explain the solution approach.\n"
    "4. List key changes file-by-file.\n"
    "5. Describe testing performed or suggested.\n"
    "6. Add reviewer notes for anything that needs special attention.\n"
    "7. Use the submit_pr_draft tool to output structured JSON."
)


def _format_pr_draft_prompt(workflow_context: dict[str, Any]) -> str:
    """将工作流上下文格式化为 PR 草稿 user prompt。"""
    parts: list[str] = []

    direction = workflow_context.get("selected_direction")
    if direction:
        parts.append(f"## Selected Direction\n{direction}")

    design = workflow_context.get("design_summary")
    if design:
        parts.append(f"## Design Summary\n{design}")

    report = workflow_context.get("implementation_report")
    if report:
        parts.append(f"## Implementation Report\n{report}")

    diff = workflow_context.get("git_diff")
    if diff:
        parts.append(f"## Git Diff\n```\n{diff[:30000]}\n```")

    changed = workflow_context.get("changed_files")
    if changed:
        if isinstance(changed, list):
            parts.append(f"## Changed Files\n" + "\n".join(f"- {f}" for f in changed))
        else:
            parts.append(f"## Changed Files\n{changed}")

    return "\n\n".join(parts) if parts else json.dumps(workflow_context, ensure_ascii=False, default=str)


def run_pr_draft_generation(
    client: Any,
    settings: Settings,
    workflow_context: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """调用 LLM 生成 PR 描述，返回结构化 PR 草稿或 None。"""
    user_prompt = _format_pr_draft_prompt(workflow_context)
    return _call_with_tool(
        client,
        settings,
        _PR_DRAFT_SYSTEM,
        user_prompt,
        _SUBMIT_PR_DRAFT_TOOL,
        "submit_pr_draft",
        repo_root=repo_root,
    )
