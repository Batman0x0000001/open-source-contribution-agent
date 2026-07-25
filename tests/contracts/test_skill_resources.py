from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.models import ToolUseBlock, ToolUseContext
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.resource_tool import ReadSkillResourceTool


def make_skill(root: Path) -> Path:
    directory = root / "method"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("""---
name: method
description: Method
when_to_use: Testing
resources: [guide.md]
input_schema: {type: object, properties: {}}
output_schema: {type: object, properties: {}}
---
Use the guide.
""", encoding="utf-8")
    return directory


def test_skill_resource_is_loaded_only_when_tool_is_called(tmp_path: Path) -> None:
    directory = make_skill(tmp_path)
    catalog = SkillCatalog([SkillLoader(tmp_path, source="project")])
    (directory / "guide.md").write_text("created after discovery", encoding="utf-8")
    registry = ToolRegistry([ReadSkillResourceTool(catalog)])
    result = asyncio.run(ToolExecutor(registry).execute(ToolUseBlock(id="r", name="read_skill_resource", input={"skill": "method", "path": "guide.md"}), ToolUseContext(session_id="s", working_directory=str(tmp_path), repository_root=str(tmp_path), state_directory=str(tmp_path / "state"))))
    assert result.error is None
    assert result.data["content"] == "created after discovery"


def test_skill_resource_rejects_undeclared_path(tmp_path: Path) -> None:
    make_skill(tmp_path)
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    catalog = SkillCatalog([SkillLoader(tmp_path, source="project")])
    registry = ToolRegistry([ReadSkillResourceTool(catalog)])
    result = asyncio.run(ToolExecutor(registry).execute(ToolUseBlock(id="r", name="read_skill_resource", input={"skill": "method", "path": "../secret.md"}), ToolUseContext(session_id="s", working_directory=str(tmp_path), repository_root=str(tmp_path), state_directory=str(tmp_path / "state"))))
    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
