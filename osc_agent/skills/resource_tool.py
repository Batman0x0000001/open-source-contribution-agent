"""向 Agent 提供受控的 Skill 资源读取工具。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from osc_agent.runtime.models import ContractModel, ToolResult, ToolUseContext, ValidationFailure, ValidationResult, ValidationSuccess
from osc_agent.runtime.tool import BaseTool
from osc_agent.skills.catalog import SkillCatalog


class ReadSkillResourceInput(ContractModel):
    skill: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ReadSkillResourceOutput(ContractModel):
    skill: str
    path: str
    content: str


class ReadSkillResourceTool(BaseTool[ReadSkillResourceInput, ReadSkillResourceOutput]):
    """只读取 Skill manifest 显式声明的延迟资源。"""

    name = "read_skill_resource"
    description = "Read a lazily loaded resource explicitly declared by a discovered Skill."
    input_model = ReadSkillResourceInput
    output_model = ReadSkillResourceOutput

    def __init__(self, catalog: SkillCatalog) -> None:
        self.catalog = catalog

    def is_read_only(self, input: ReadSkillResourceInput) -> bool:
        return True

    def is_concurrency_safe(self, input: ReadSkillResourceInput) -> bool:
        return True

    async def validate_input(self, input: ReadSkillResourceInput, context: ToolUseContext) -> ValidationResult:
        descriptor = self.catalog.get(input.skill)
        if descriptor is None:
            return ValidationFailure(reason="skill not found")
        if input.path not in descriptor.manifest.resources:
            return ValidationFailure(reason="resource is not declared by the skill manifest")
        root = Path(descriptor.root).resolve()
        target = (root / input.path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            return ValidationFailure(reason="skill resource is missing or escapes the skill root")
        return ValidationSuccess()

    async def call(self, input: ReadSkillResourceInput, context: ToolUseContext) -> ToolResult:
        descriptor = self.catalog.get(input.skill)
        assert descriptor is not None
        target = (Path(descriptor.root) / input.path).resolve()
        return ToolResult(data={"skill": input.skill, "path": input.path, "content": target.read_text(encoding="utf-8")})
