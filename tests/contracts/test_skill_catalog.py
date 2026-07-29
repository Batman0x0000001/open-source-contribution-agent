"""验证 Skill 来源优先级、可见性和工具声明。"""

from __future__ import annotations

from pathlib import Path

import pytest

from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.loader import SkillLoader


def _write_skill(
    root: Path,
    name: str,
    body: str,
    *,
    flags: str = "",
    allowed_tools: str = "[read_file]",
) -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
name: {name}
description: {name} description
when_to_use: Use {name} for tests
allowed_tools: {allowed_tools}
{flags}---
{body}
""",
        encoding="utf-8",
    )
    return path


def test_builtin_names_are_reserved_and_project_overrides_user(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    builtin_path = _write_skill(builtin, "reserved", "builtin")
    _write_skill(user, "reserved", "user")
    _write_skill(project, "reserved", "project")
    _write_skill(user, "custom", "user")
    project_path = _write_skill(project, "custom", "project")

    catalog = SkillCatalog(
        [
            SkillLoader(project, source="project"),
            SkillLoader(builtin, source="builtin"),
            SkillLoader(user, source="user"),
        ]
    )

    assert catalog.get("reserved").path == str(builtin_path.resolve())
    assert catalog.get("custom").path == str(project_path.resolve())
    assert [item.code for item in catalog.diagnostics()] == [
        "BUILTIN_SKILL_NAME_RESERVED",
        "BUILTIN_SKILL_NAME_RESERVED",
    ]


def test_catalog_is_the_only_source_of_user_and_model_visibility(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    project = tmp_path / "project"
    _write_skill(builtin, "shared", "shared")
    _write_skill(
        builtin,
        "hidden",
        "hidden",
        flags="user_invocable: false\ndisable_model_invocation: true\n",
    )
    _write_skill(project, "external", "external")
    catalog = SkillCatalog(
        [SkillLoader(builtin, source="builtin"), SkillLoader(project, source="project")]
    )

    assert [item.manifest.name for item in catalog.list_user_invocable()] == [
        "external",
        "shared",
    ]
    assert [item.manifest.name for item in catalog.list_model_invocable()] == ["shared"]
    assert catalog.resolve("hidden", "user") is None
    assert catalog.resolve("hidden", "model") is None
    assert catalog.resolve("hidden", "product").manifest.name == "hidden"


def test_unknown_external_tool_skips_skill_but_builtin_fails(tmp_path: Path) -> None:
    external = tmp_path / "external"
    _write_skill(external, "external", "external", allowed_tools="[missing]")
    catalog = SkillCatalog([SkillLoader(external, source="project")])

    catalog.validate_allowed_tools({"read_file"})

    assert catalog.get("external") is None
    assert catalog.diagnostics()[0].code == "UNKNOWN_SKILL_TOOL"

    builtin = tmp_path / "builtin"
    _write_skill(builtin, "builtin", "builtin", allowed_tools="[missing]")
    catalog = SkillCatalog([SkillLoader(builtin, source="builtin")])
    with pytest.raises(ValueError, match="unknown allowed_tools"):
        catalog.validate_allowed_tools({"read_file"})
