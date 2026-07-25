from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.agents.definitions import AgentDefinition, AgentInvocation, AgentRunResult
from osc_agent.runtime.models import CapabilityScope, RuntimeMessage, TextBlock, ToolUseContext
from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.executor import SkillExecutor
from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.models import SkillInvocation
from osc_agent.skills.tool import SkillTool, SkillToolInput


def write_skill(
    root: Path,
    *,
    name: str = "review",
    source_text: str = "Review carefully.",
    context: str = "inline",
    allowed_tools: str = "[read, write]",
) -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
name: {name}
version: 1
description: Review a change
when_to_use: Before submitting a change
allowed_tools: {allowed_tools}
context: {context}
input_schema:
  type: object
  properties:
    target:
      type: string
  required: [target]
output_schema:
  type: object
  properties:
    verdict:
      type: string
  required: [verdict]
---
{source_text}
""",
        encoding="utf-8",
    )
    return path


def invoke(name: str = "review", *, arguments=None, parent_messages=None) -> SkillInvocation:
    return SkillInvocation(
        name=name,
        arguments=arguments or {"target": "change.py"},
        session_id="parent-1",
        working_directory="C:/repo",
        caller_capabilities=CapabilityScope(allowed_tools=frozenset({"read"})),
        parent_messages=parent_messages or [],
    )


def test_catalog_keeps_body_lazy_until_invocation(tmp_path: Path) -> None:
    skill_path = write_skill(tmp_path, source_text="Original body.")
    catalog = SkillCatalog([SkillLoader(tmp_path, source="project")])
    raw = skill_path.read_text(encoding="utf-8").replace("Original body.", "Updated after discovery.")
    skill_path.write_text(raw, encoding="utf-8")

    result = asyncio.run(SkillExecutor(catalog).execute(invoke()))

    assert catalog.get("review").manifest.description == "Review a change"
    assert "Updated after discovery." in result.rendered_prompt
    assert "Original body." not in result.rendered_prompt


def test_project_skill_overrides_user_and_builtin_sources(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    write_skill(builtin, source_text="builtin")
    write_skill(user, source_text="user")
    project_path = write_skill(project, source_text="project")

    catalog = SkillCatalog(
        [
            SkillLoader(project, source="project"),
            SkillLoader(builtin, source="builtin"),
            SkillLoader(user, source="user"),
        ]
    )

    assert catalog.get("review").path == str(project_path.resolve())


def test_skill_input_is_strictly_validated_by_generated_pydantic_model(tmp_path: Path) -> None:
    write_skill(tmp_path)
    executor = SkillExecutor(SkillCatalog([SkillLoader(tmp_path, source="project")]))

    result = asyncio.run(executor.execute(invoke(arguments={"target": 123})))

    assert result.status == "failed"
    assert "string_type" in result.error


def test_fork_skill_uses_agent_runner_and_validates_structured_output(tmp_path: Path) -> None:
    write_skill(tmp_path, context="fork")

    class RecordingRunner:
        def __init__(self) -> None:
            self.definition: AgentDefinition | None = None
            self.invocation: AgentInvocation | None = None

        async def run_with_definition(
            self,
            definition: AgentDefinition,
            invocation: AgentInvocation,
        ) -> AgentRunResult:
            self.definition = definition
            self.invocation = invocation
            return AgentRunResult(
                task_id="task-1",
                session_id="child-1",
                status="completed",
                output='{"verdict":"pass"}',
            )

    runner = RecordingRunner()
    executor = SkillExecutor(
        SkillCatalog([SkillLoader(tmp_path, source="project")]),
        agent_runner=runner,  # type: ignore[arg-type]
    )
    parent = [RuntimeMessage(role="user", content=[TextBlock(text="parent")])]

    result = asyncio.run(executor.execute(invoke(parent_messages=parent)))

    assert result.status == "completed"
    assert result.output == {"verdict": "pass"}
    assert runner.definition.capabilities.allowed_tools == frozenset({"read"})
    assert runner.invocation.mode == "fork"


def test_skill_tool_delegates_to_the_shared_executor(tmp_path: Path) -> None:
    write_skill(tmp_path)
    executor = SkillExecutor(SkillCatalog([SkillLoader(tmp_path, source="project")]))
    tool = SkillTool(executor)
    context = ToolUseContext(session_id="session-1", working_directory="C:/repo", repository_root="C:/repo", state_directory="C:/state")

    result = asyncio.run(
        tool.call(SkillToolInput(skill="review", arguments={"target": "change.py"}), context)
    )

    assert result.data["status"] == "inline"
    assert len(result.new_messages) == 1
    assert "Review carefully." in result.new_messages[0].content[0].text
