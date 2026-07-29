"""验证模型 Skill Tool 的 wire contract 和 Runtime 上下文更新。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import CapabilityScope, ToolUseContext
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.runtime.tool_orchestration import run_tools
from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.invocation_tool import SkillTool, SkillToolInput, SkillToolOutput
from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.preparer import SkillPreparer


def _write_skill(root: Path, name: str, *, source: str = "builtin") -> None:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
name: {name}
description: Review a change
when_to_use: Before submitting a change
allowed_tools: [read_file]
completion:
  required_evidence: [successful_test, git_change_snapshot]
---
Review carefully.
""",
        encoding="utf-8",
    )


def test_skill_tool_delegates_to_preparer_and_updates_context(tmp_path: Path) -> None:
    _write_skill(tmp_path, "review")
    preparer = SkillPreparer(SkillCatalog([SkillLoader(tmp_path, source="builtin")]))
    tool = SkillTool(preparer)
    context = ToolUseContext(
        session_id="session-1",
        working_directory=str(tmp_path),
        state_directory=str(tmp_path / "state"),
        capabilities=CapabilityScope(allowed_tools=frozenset({"skill", "read_file", "grep"})),
    )

    result = asyncio.run(
        tool.call(SkillToolInput(skill="review", arguments={"target": 123}), context)
    )

    assert result.data == {
        "skill": "review",
        "status": "inline",
        "output": None,
        "error": None,
    }
    assert "Review carefully." in result.new_messages[0].content[0].text
    assert result.context_update.capabilities.allowed_tools == {"read_file"}

    registry = ToolRegistry([tool])

    async def execute_through_runtime_path():
        return [
            update
            async for update in run_tools(
                [ToolUseBlock(id="skill-call", name="skill", input={"skill": "review"})],
                executor=ToolExecutor(registry),
                context=context,
            )
        ]

    updates = asyncio.run(execute_through_runtime_path())
    assert updates[-1].context.completion_requirements.required_evidence == {
        "successful_test",
        "git_change_snapshot",
    }


def test_skill_tool_description_uses_catalog_model_visibility(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    project = tmp_path / "project"
    _write_skill(builtin, "builtin-review")
    _write_skill(project, "project-review")
    tool = SkillTool(
        SkillPreparer(
            SkillCatalog(
                [
                    SkillLoader(builtin, source="builtin"),
                    SkillLoader(project, source="project"),
                ]
            )
        )
    )

    assert "builtin-review" in tool.description
    assert "Before submitting a change" in tool.description
    assert "project-review" not in tool.description


def test_skill_tool_keeps_the_existing_status_wire_contract() -> None:
    status_schema = SkillToolOutput.model_json_schema()["properties"]["status"]

    assert status_schema["enum"] == ["inline", "completed", "failed"]
