"""
收集 repo + messages 状态
    ↓
构建 PromptContext（统一状态对象）
    ↓
注入 memory / todos / git / skills / tools
    ↓
拼接 system prompt 各模块
    ↓
缓存 prompt（避免重复构建）
    ↓
输出最终 system prompt → LLM
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from osc_agent.harness.memory import memory_prompt
from osc_agent.harness.todo import current_todos
from osc_agent.skills.registry import list_skill_catalog, suggest_skills_for_repo
from osc_agent.tools.git import git_status
from osc_agent.tools.repo import inspect_repo


@dataclass(frozen=True)
class PromptContext:
    repo_root: str
    current_task: str
    enabled_tools: list[str]
    repo_overview: str
    permissions: str
    skill_catalog: str
    suggested_skills: list[str]
    memory: str
    current_todos: list[dict[str, str]]
    git_state: str


PROMPT_SECTIONS = {
    "identity": (
        "You are a coding agent. Produce reviewable open source contributions, keep changes scoped, "
        "and explain results clearly."
    ),
    "repo": "Repository context:\n{repo_overview}",
    "task": "Current task:\n{current_task}",
    "tools": "Enabled tools:\n{enabled_tools}",
    "permissions": "Permission boundaries:\n{permissions}",
    "skills": (
        "Skills available:\n{skill_catalog}\n\n"
        "Suggested skills for this repository: {suggested_skills}.\n"
        "Use load_skill(name) to load full skill instructions only when they are relevant. "
        "Do not assume the catalog contains the full instructions."
    ),
    "memory": "Persistent memory:\n{memory}",
    "current_todos": "Current todos:\n{current_todos}",
    "git_state": "Git state:\n{git_state}",
}

_last_context_key: str | None = None
_last_prompt: str | None = None


def update_context(
    *,
    repo_root: Path,
    messages: list[dict[str, Any]],
    enabled_tools: list[str],
) -> PromptContext:
    """从真实运行状态构造 prompt context，避免根据关键词猜测可用能力。"""
    current_task = _latest_user_text(messages)
    catalog = list_skill_catalog()
    suggestions = suggest_skills_for_repo(repo_root)
    return PromptContext(
        repo_root=str(repo_root.resolve()),
        current_task=current_task or "(no explicit task in current messages)",
        enabled_tools=sorted(enabled_tools),
        repo_overview=inspect_repo(repo_root=repo_root),
        permissions=_permission_summary(),
        skill_catalog=catalog,
        suggested_skills=suggestions,
        memory=memory_prompt(repo_root, query=current_task),
        current_todos=current_todos(),
        git_state=git_status(repo_root=repo_root),
    )


def assemble_system_prompt(context: PromptContext | Path, *, current_task: str = "") -> str:
    """按 section 组装 system prompt；兼容旧调用传入 repo_root 的方式。"""
    if isinstance(context, Path):
        context = update_context(repo_root=context, messages=[{"role": "user", "content": current_task}], enabled_tools=[])

    rendered = [
        PROMPT_SECTIONS["identity"],
        PROMPT_SECTIONS["repo"].format(repo_overview=context.repo_overview),
        PROMPT_SECTIONS["task"].format(current_task=context.current_task),
        PROMPT_SECTIONS["tools"].format(enabled_tools=", ".join(context.enabled_tools) or "(none)"),
        PROMPT_SECTIONS["permissions"].format(permissions=context.permissions),
        PROMPT_SECTIONS["skills"].format(
            skill_catalog=context.skill_catalog,
            suggested_skills=", ".join(context.suggested_skills) if context.suggested_skills else "(none)",
        ),
        PROMPT_SECTIONS["memory"].format(memory=context.memory),
        PROMPT_SECTIONS["current_todos"].format(current_todos=_format_todos(context.current_todos)),
        PROMPT_SECTIONS["git_state"].format(git_state=context.git_state),
        (
            "Contribution rule: before modifying files, call todo_write with a plan that covers understanding "
            "the task, reading contribution guidance, locating files, editing, testing, and drafting the PR. "
            "Do not run git push or open pull requests automatically."
        ),
    ]
    return "\n\n".join(rendered)


def get_system_prompt(context: PromptContext) -> str:
    """用稳定 JSON key 缓存拼接结果；这里只缓存字符串组装，不代表 API prompt cache。"""
    global _last_context_key, _last_prompt
    key = json.dumps(asdict(context), ensure_ascii=False, sort_keys=True, default=str)
    if key == _last_context_key and _last_prompt is not None:
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            summaries: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    summaries.append(f"tool_result {block.get('tool_use_id')}: {str(block.get('content', ''))[:200]}")
            if summaries:
                return "\n".join(summaries)
    return ""


def _format_todos(todos: list[dict[str, str]]) -> str:
    if not todos:
        return "(no todos)"
    return "\n".join(f"- {todo['status']}: {todo['content']}" for todo in todos)


def _permission_summary() -> str:
    return (
        "File tools are limited to the target repository. Dangerous shell commands such as git push, "
        "gh pr create, sudo, shutdown, reboot, mkfs, and dd if= are denied. Destructive or dependency-changing "
        "commands require explicit confirmation."
    )
