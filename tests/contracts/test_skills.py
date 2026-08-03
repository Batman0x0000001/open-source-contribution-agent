"""验证项目随包发布的内置 Skill 保持纯声明式方法边界。"""

from __future__ import annotations

from pathlib import Path

from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.loader import SkillLoader


def test_builtin_skills_are_discovered_from_dedicated_content_directory() -> None:
    root = Path(__file__).parents[2] / "osc_agent" / "skills" / "builtins"
    catalog = SkillCatalog([SkillLoader(root, source="builtin")])

    assert [item.manifest.name for item in catalog.list()] == [
        "issue-planning",
        "open-source-contribution",
    ]
    assert catalog.diagnostics() == []


def test_skill_framework_has_no_schema_or_fork_execution_path() -> None:
    root = Path(__file__).parents[2] / "osc_agent" / "skills"
    python_source = "\n".join(
        path.read_text(encoding="utf-8") for path in root.glob("*.py")
    )
    manifests = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "builtins").rglob("SKILL.md")
    )

    assert "SubagentRunner" not in python_source
    assert "create_model" not in python_source
    for removed in ("input_schema:", "output_schema:", "context: fork", "context: inline"):
        assert removed not in manifests
