"""将 Skill 调用封装为主 Runtime 可使用的 skill 工具。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from osc_agent.runtime.models import ContextUpdate, ContractModel, RuntimeMessage, TextBlock, ToolResult, ToolUseContext
from osc_agent.runtime.tool import BaseTool
from osc_agent.skills.executor import SkillExecutor
from osc_agent.skills.models import SkillInvocation


class SkillToolInput(ContractModel):
    skill: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class SkillToolOutput(ContractModel):
    skill: str
    status: Literal["inline", "completed", "failed"]
    output: JsonValue = None
    error: str | None = None


class SkillTool(BaseTool[SkillToolInput, SkillToolOutput]):
    name = "skill"
    input_model = SkillToolInput
    output_model = SkillToolOutput

    def __init__(self, executor: SkillExecutor) -> None:
        self.executor = executor

    @property
    def description(self) -> str:
        available = [
            f"{item.manifest.name}: {item.manifest.description}"
            for item in self.executor.catalog.list()
            if item.source == "builtin" and not item.manifest.disable_model_invocation
        ]
        suffix = "; ".join(available) if available else "none"
        return f"Invoke one model-available Skill through the shared SkillExecutor. Available: {suffix}"

    async def call(self, input: SkillToolInput, context: ToolUseContext) -> ToolResult:
        result = await self.executor.execute(
            SkillInvocation(
                name=input.skill,
                arguments=input.arguments,
                session_id=context.session_id,
                working_directory=context.working_directory,
                caller_capabilities=context.capabilities,
                parent_messages=context.transcript_messages,
                trigger="model",
            )
        )
        new_messages = (
            [RuntimeMessage(role="user", content=[TextBlock(text=result.rendered_prompt)])]
            if result.status == "inline"
            else []
        )
        return ToolResult(
            data={
                "skill": result.name,
                "status": result.status,
                "output": result.output,
                "error": result.error,
            },
            new_messages=new_messages,
            context_update=(
                ContextUpdate(
                    capabilities=result.capabilities,
                    completion_requirements=result.completion_requirements,
                )
                if result.status == "inline" and result.capabilities is not None
                else None
            ),
        )
