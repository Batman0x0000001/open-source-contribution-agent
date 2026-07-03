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
        name = str(meta.get("name") or directory.name).strip()
        description = str(meta.get("description") or first_heading(raw) or name).strip()
        if name:
            registry[name] = Skill(name=name, description=description, content=raw, path=manifest)
    return registry


def list_skill_catalog(skills_root: Path | None = None) -> str:
    skills = scan_skills(skills_root)
    if not skills:
        return "(no skills available)"
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills.values())


def load_skill(name: str, *, skills_root: Path | None = None) -> str:
    """按注册名加载技能正文，避免让模型传文件路径造成路径遍历。"""
    skills = scan_skills(skills_root)
    skill = skills.get(name)
    if skill is None:
        return f"Skill not found: {name}"
    return skill.content


def suggest_skills_for_repo(repo_root: Path, *, skills_root: Path | None = None) -> list[str]:
    available = set(scan_skills(skills_root))
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

    return sorted(suggestion for suggestion in suggestions if suggestion in available)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        return {}, raw

    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw

    meta: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip("\"'")
    body = raw[end + len("\n---") :].lstrip()
    return meta, body


def first_heading(raw: str) -> str | None:
    for line in raw.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None
