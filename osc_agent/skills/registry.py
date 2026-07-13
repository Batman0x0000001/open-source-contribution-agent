"""
扫描 skills 目录
    ↓
查找每个子目录里的 SKILL.md
    ↓
解析 name / description
    ↓
注册成 Skill
    ↓
LLM 可通过 load_skill(name) 加载完整技能内容
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


LOAD_SKILL_TOOL = {
    "name": "load_skill",
    "description": "Load the full instructions for a named contribution skill.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: Path
    schema_version: int
    version: str
    applies_to: tuple[str, ...]
    required_tools: tuple[str, ...]
    permissions: tuple[str, ...]
    input_contract: str
    output_contract: str


REQUIRED_SKILL_FIELDS = {
    "schema_version",
    "name",
    "description",
    "version",
    "applies_to",
    "required_tools",
    "permissions",
    "input_contract",
    "output_contract",
}
DEFAULT_AVAILABLE_TOOLS = {"read_file", "write_file", "edit_file", "glob", "bash", "git_status", "git_diff"}
DEFAULT_PERMISSIONS = {"read", "write", "shell"}


def default_skills_root() -> Path:
    return Path(__file__).resolve().parent


def scan_skills(skills_root: Path | None = None) -> dict[str, Skill]:
    root = (skills_root or default_skills_root()).resolve()
    registry: dict[str, Skill] = {}
    if not root.exists():
        return registry

    #遍历根目录下的所有子项
    for directory in sorted(root.iterdir()):
        manifest = directory / "SKILL.md"
        if not directory.is_dir() or not manifest.exists():
            continue
        raw = manifest.read_text(encoding="utf-8")
        meta, _body = parse_frontmatter(raw)
        missing = sorted(REQUIRED_SKILL_FIELDS - set(meta))
        if missing:
            raise ValueError(f"invalid skill {manifest}: missing fields: {', '.join(missing)}")
        if meta.get("schema_version") != 1:
            raise ValueError(f"invalid skill {manifest}: schema_version must be 1")
        name = str(meta["name"]).strip()
        description = str(meta["description"]).strip()
        list_fields = ("applies_to", "required_tools", "permissions")
        for field_name in list_fields:
            if not isinstance(meta[field_name], list) or not all(isinstance(item, str) for item in meta[field_name]):
                raise ValueError(f"invalid skill {manifest}: {field_name} must be a list of strings")
        registry[name] = Skill(
            name=name,
            description=description,
            content=raw,
            path=manifest,
            schema_version=1,
            version=str(meta["version"]),
            applies_to=tuple(meta["applies_to"]),
            required_tools=tuple(meta["required_tools"]),
            permissions=tuple(meta["permissions"]),
            input_contract=str(meta["input_contract"]),
            output_contract=str(meta["output_contract"]),
        )
    return registry


def list_skill_catalog(skills_root: Path | None = None) -> str:
    skills = scan_skills(skills_root)
    if not skills:
        return "(no skills available)"
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills.values())


def load_skill(
    name: str,
    *,
    skills_root: Path | None = None,
    available_tools: set[str] | None = None,
    granted_permissions: set[str] | None = None,
) -> str:
    """按注册名加载技能正文，避免让模型传文件路径造成路径遍历。"""
    skills = scan_skills(skills_root)
    skill = skills.get(name)
    if skill is None:
        return f"Skill not found: {name}"
    active_tools = DEFAULT_AVAILABLE_TOOLS if available_tools is None else available_tools
    active_permissions = DEFAULT_PERMISSIONS if granted_permissions is None else granted_permissions
    missing_tools = set(skill.required_tools) - active_tools
    if missing_tools:
        return f"Skill unavailable: missing tools: {', '.join(sorted(missing_tools))}"
    missing_permissions = set(skill.permissions) - active_permissions
    if missing_permissions:
        return f"Skill unavailable: permissions not granted: {', '.join(sorted(missing_permissions))}"
    return skill.content


def suggest_skills_for_repo(
    repo_root: Path,
    *,
    skills_root: Path | None = None,
    available_tools: set[str] | None = None,
    granted_permissions: set[str] | None = None,
) -> list[str]:
    skills = scan_skills(skills_root)
    available = set(skills)
    suggestions: set[str] = {"open-source"}

    markers = {path.name.lower() for path in repo_root.iterdir()} if repo_root.exists() else set()
    if {"pyproject.toml", "requirements.txt", "setup.py"} & markers or any(repo_root.glob("*.py")):
        suggestions.add("python")
    if {"package.json", "tsconfig.json"} & markers:
        suggestions.add("javascript")
    if any(repo_root.glob("README*")) or any(repo_root.glob("docs")):
        suggestions.add("docs")
    if any((repo_root / name).exists() for name in ("tests", "test")):
        suggestions.add("tests")

    tools = DEFAULT_AVAILABLE_TOOLS if available_tools is None else available_tools
    permissions = DEFAULT_PERMISSIONS if granted_permissions is None else granted_permissions
    return sorted(
        suggestion for suggestion in suggestions
        if suggestion in available
        and set(skills[suggestion].required_tools) <= tools
        and set(skills[suggestion].permissions) <= permissions
    )


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    if not raw.startswith("---\n"):
        return {}, raw

    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw

    parsed = yaml.safe_load(raw[4:end]) or {}
    if not isinstance(parsed, dict):
        raise ValueError("skill frontmatter must be a YAML object")
    meta: dict[str, Any] = parsed
    body = raw[end + len("\n---") :].lstrip()
    return meta, body


def first_heading(raw: str) -> str | None:
    for line in raw.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None
