"""
收集 repo + 运行时状态
    ↓
构建 PromptContext（统一状态对象）
    ↓
注入 capability / risk / memory / todos / git / skills / tools
    ↓
拼接 system prompt 各模块
    ↓
输出最终 system prompt → LLM
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osc_agent.harness.capabilities import AgentCapabilityScope, AgentExecutionStage
from osc_agent.harness.memory import memory_prompt
from osc_agent.harness.repository_boundary import repository_boundary_summary
from osc_agent.harness.risk import risk_policy_summary
from osc_agent.harness.runtime_state import refresh_runtime_state
from osc_agent.harness.todo import current_todos
from osc_agent.skills.registry import list_skill_catalog, suggest_skills_for_repo
from osc_agent.tools.git import git_status
from osc_agent.tools.repo import inspect_repo


@dataclass(frozen=True)
class PromptContext:
    objective: str
    current_instruction: str
    enabled_tools: list[str]
    repo_overview: str
    capabilities: AgentCapabilityScope
    boundary_controls: str
    risk_controls: str
    skill_catalog: str
    suggested_skills: list[str]
    memory: str
    current_todos: list[dict[str, str]]
    git_state: str
    runtime_state: dict[str, Any]


PROMPT_SECTIONS = {
    "identity": (
        "You are a coding agent. Produce reviewable open source contributions, keep changes scoped, "
        "and explain results clearly."
    ),
    "repo": "Repository context:\n{repo_overview}",
    "objective": "Top-level objective:\n{objective}",
    "instruction": "Current execution instruction:\n{current_instruction}",
    "effective_boundary": "Effective execution boundary:\n{effective_boundary}",
    "repository_boundary": "Repository boundary:\n{boundary_controls}",
    "risk": "Risk controls:\n{risk_controls}",
    "skills": (
        "Skills available:\n{skill_catalog}\n\n"
        "Suggested skills for this repository: {suggested_skills}.\n"
        "Use load_skill(name) to load full skill instructions only when they are relevant. "
        "Do not assume the catalog contains the full instructions."
    ),
    "memory": "Persistent memory:\n{memory}",
    "current_todos": "Current todos:\n{current_todos}",
    "git_state": "Git state:\n{git_state}",
    "runtime_state": "Authoritative runtime state (never compacted):\n{runtime_state}",
}

SYSTEM_PROMPT_CHAR_LIMIT = 32_000
SECTION_CHAR_LIMITS = {
    "objective": 4_000,
    "instruction": 8_000,
    "effective_boundary": 5_000,
    "repository_boundary": 1_000,
    "risk": 3_000,
    "repo": 6_000,
    "runtime_state": 9_000,
    "current_todos": 3_000,
    "git_state": 4_000,
    "memory": 4_000,
    "skills": 5_000,
}


def update_context(
    *,
    repo_root: Path,
    objective: str,
    current_instruction: str,
    enabled_tools: list[str],
    capabilities: AgentCapabilityScope,
    run_id: str | None = None,
    session_id: str = "default",
) -> PromptContext:
    """从真实运行状态构造 PromptContext，不从任务文本猜测阶段或能力。"""
    enabled = sorted(enabled_tools)
    disallowed = [tool for tool in enabled if not capabilities.permits_tool(tool)]
    if disallowed:
        raise ValueError(
            "enabled_tools exceed the capability scope: " + ", ".join(disallowed)
        )
    skills_enabled = "load_skill" in enabled
    catalog = list_skill_catalog() if skills_enabled else ""
    suggestions = suggest_skills_for_repo(repo_root) if skills_enabled else []
    runtime_state = refresh_runtime_state(
        repo_root,
        current_instruction,
        objective=objective,
        run_id=run_id,
        session_id=session_id,
    )
    return PromptContext(
        objective=runtime_state.objective or "(no explicit top-level objective)",
        current_instruction=(
            runtime_state.current_instruction or "(no current execution instruction)"
        ),
        enabled_tools=enabled,
        repo_overview=inspect_repo(repo_root=repo_root),
        capabilities=capabilities,
        boundary_controls=repository_boundary_summary(),
        risk_controls=risk_policy_summary(),
        skill_catalog=catalog,
        suggested_skills=suggestions,
        memory=memory_prompt(repo_root, query=objective or current_instruction),
        current_todos=current_todos(repo_root),
        git_state=git_status(repo_root=repo_root),
        runtime_state=runtime_state.model_dump(mode="json"),
    )


def assemble_system_prompt(context: PromptContext) -> str:
    """按当前能力、阶段和运行状态组装 system prompt。"""
    rendered = [
        PROMPT_SECTIONS["identity"],
        _bounded_section(
            "objective",
            PROMPT_SECTIONS["objective"].format(objective=context.objective),
        ),
        _bounded_section(
            "instruction",
            PROMPT_SECTIONS["instruction"].format(
                current_instruction=context.current_instruction
            ),
        ),
        _bounded_section(
            "effective_boundary",
            PROMPT_SECTIONS["effective_boundary"].format(
                effective_boundary=_format_effective_boundary(context)
            ),
        ),
        _bounded_section(
            "repository_boundary",
            PROMPT_SECTIONS["repository_boundary"].format(
                boundary_controls=context.boundary_controls
            ),
        ),
        _bounded_section(
            "risk",
            PROMPT_SECTIONS["risk"].format(risk_controls=context.risk_controls),
        ),
        *_operating_rules(context),
        _bounded_section(
            "runtime_state",
            PROMPT_SECTIONS["runtime_state"].format(
                runtime_state=json.dumps(
                    _runtime_state_for_prompt(context.runtime_state),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            ),
        ),
        _bounded_section(
            "repo",
            PROMPT_SECTIONS["repo"].format(repo_overview=context.repo_overview),
        ),
        _bounded_section(
            "current_todos",
            PROMPT_SECTIONS["current_todos"].format(
                current_todos=_format_todos(context.current_todos)
            ),
        ),
        _bounded_section(
            "git_state",
            PROMPT_SECTIONS["git_state"].format(git_state=context.git_state),
        ),
        _bounded_section(
            "memory",
            PROMPT_SECTIONS["memory"].format(memory=context.memory),
        ),
    ]
    if "load_skill" in context.enabled_tools:
        rendered.append(
            _bounded_section(
                "skills",
                PROMPT_SECTIONS["skills"].format(
                    skill_catalog=context.skill_catalog,
                    suggested_skills=(
                        ", ".join(context.suggested_skills)
                        if context.suggested_skills
                        else "(none)"
                    ),
                ),
            )
        )
    return _join_with_budget(rendered)


def _format_todos(todos: list[dict[str, str]]) -> str:
    if not todos:
        return "(no todos)"
    lines: list[str] = []
    for todo in todos:
        if not isinstance(todo, dict):
            lines.append("- unknown: (malformed todo)")
            continue
        status = str(todo.get("status") or "unknown")
        content = str(todo.get("content") or "(missing content)")
        lines.append(f"- {status}: {content}")
    return "\n".join(lines)


def _format_effective_boundary(context: PromptContext) -> str:
    capabilities = context.capabilities
    unrestricted = capabilities.allowed_tools is None
    values = {
        "stage": capabilities.stage.value,
        "capability_mode": "unrestricted" if unrestricted else "restricted",
        "effective_tools": context.enabled_tools,
        "writable_paths": (
            ["repository/**"] if unrestricted else list(capabilities.writable_paths)
        ),
        "forbidden_paths": list(capabilities.forbidden_paths),
        "note": (
            "unrestricted applies only to capability filtering; repository and risk controls remain active"
            if unrestricted
            else "effective_tools are the registered tools permitted by this capability scope"
        ),
    }
    return json.dumps(values, ensure_ascii=False, indent=2)


def _runtime_state_for_prompt(runtime_state: dict[str, Any]) -> dict[str, Any]:
    duplicated = {"objective", "current_instruction", "allowed_files", "forbidden_paths"}
    return {key: value for key, value in runtime_state.items() if key not in duplicated}


def _operating_rules(context: PromptContext) -> list[str]:
    rules: list[str] = []
    if "todo_write" in context.enabled_tools:
        rules.append(
            "Planning rule: before modifying files, call todo_write with a concise plan covering understanding, "
            "editing, and verification. Keep todo status current while working."
        )
    rules.append(_stage_rule(context.capabilities.stage))
    return rules


def _stage_rule(stage: AgentExecutionStage) -> str:
    rules = {
        AgentExecutionStage.UNDERSTANDING: (
            "Stage rule: read files and summarize scope only. Do not edit files. Return only the exact JSON "
            "Understanding checkpoint requested by the user prompt; decision must be READY_TO_EDIT or "
            "CONTRACT_UPDATE_REQUIRED."
        ),
        AgentExecutionStage.REPRODUCE: (
            "Stage rule: create the smallest regression test using only writable_paths. Do not modify production "
            "code, configuration, or unrelated tests."
        ),
        AgentExecutionStage.EDIT: (
            "Stage rule: modify only writable_paths, preserve forbidden_paths, and keep the implementation "
            "focused on the saved design."
        ),
        AgentExecutionStage.REPAIR: (
            "Stage rule: inspect the recorded verification failure and make the smallest repair within "
            "writable_paths. Do not weaken or modify frozen regression tests."
        ),
        AgentExecutionStage.VERIFICATION: (
            "Stage rule: inspect the current diff and verification evidence without modifying files. Report exact "
            "commands and results already available in runtime state."
        ),
    }
    return rules.get(stage, "Stage rule: no specialized contribution stage is active.")


def _bounded_section(name: str, text: str) -> str:
    return _truncate(text, SECTION_CHAR_LIMITS[name])


def _join_with_budget(sections: list[str]) -> str:
    rendered: list[str] = []
    used = 0
    for section in sections:
        separator_length = 2 if rendered else 0
        remaining = SYSTEM_PROMPT_CHAR_LIMIT - used - separator_length
        if remaining <= 0:
            break
        clipped = _truncate(section, remaining)
        rendered.append(clipped)
        used += separator_length + len(clipped)
        if len(clipped) < len(section):
            break
    return "\n\n".join(rendered)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]"
    if limit <= len(marker):
        return text[:limit]
    return text[: limit - len(marker)] + marker
