from __future__ import annotations

from pathlib import Path

from osc_agent.skills.registry import list_skill_catalog, suggest_skills_for_repo


def assemble_system_prompt(repo_root: Path) -> str:
    """只注入技能目录和建议，不把完整 SKILL.md 正文塞进 system prompt。"""
    catalog = list_skill_catalog()
    suggestions = suggest_skills_for_repo(repo_root)
    suggested_text = ", ".join(suggestions) if suggestions else "(none)"
    return (
        f"You are a coding agent working inside this local repository: {repo_root}. "
        "Use the repo, file, git, bash, todo, task, and skill tools to solve the user's contribution task. "
        "Before modifying files for a contribution, call todo_write with a contribution plan that covers "
        "understanding the task, reading contribution guidance, locating files, editing, testing, and drafting the PR.\n\n"
        "Skills available:\n"
        f"{catalog}\n\n"
        f"Suggested skills for this repository: {suggested_text}.\n"
        "Use load_skill(name) to load full skill instructions only when they are relevant. "
        "Do not assume the catalog contains the full instructions."
    )
