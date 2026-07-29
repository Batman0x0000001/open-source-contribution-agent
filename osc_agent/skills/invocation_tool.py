"""将模型的 Skill Tool 调用适配为当前 Conversation 的方法注入。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import RuntimeMessage, TextBlock
from osc_agent.runtime.state import (
    CapabilitiesRestricted,
    CompletionTightened,
    ToolContext,
)
from osc_agent.runtime.tool_models import ToolResult
from osc_agent.runtime.tool import BaseTool
from osc_agent.skills.models import PreparedSkill, SkillRequest
from osc_agent.skills.preparer import SkillPreparer


class SkillToolInput(ContractModel):
    skill: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class SkillToolOutput(ContractModel):
    skill: str
    # completed 保留在 Tool wire contract 中，当前 inline-only 实现不会再生成该状态。
    status: Literal["inline", "completed", "failed"]
    output: JsonValue = None
    error: str | None = None


class SkillTool(BaseTool[SkillToolInput, SkillToolOutput]):
    name = "skill"
    input_model = SkillToolInput
    output_model = SkillToolOutput

    def __init__(self, preparer: SkillPreparer) -> None:
        self.preparer = preparer

    @property
    def description(self) -> str:
        available = [
            f"{item.manifest.name}: {item.manifest.description}. {item.manifest.when_to_use}"
            for item in self.preparer.catalog.list_model_invocable()
        ]
        suffix = "; ".join(available) if available else "none"
        return f"Invoke one model-available Skill in the current conversation. Available: {suffix}"

    async def call(self, input: SkillToolInput, context: ToolContext) -> ToolResult:
        result = await self.preparer.prepare(
            SkillRequest(
                name=input.skill,
                arguments=input.arguments,
                caller_capabilities=context.capabilities,
                trigger="model",
            )
        )
        if not isinstance(result, PreparedSkill):
            return ToolResult(
                data={
                    "skill": result.name,
                    "status": "failed",
                    "output": None,
                    "error": result.error,
                }
            )
        return ToolResult(
            data={
                "skill": result.name,
                "status": "inline",
                "output": None,
                "error": None,
            },
            new_messages=[
                RuntimeMessage(role="user", content=[TextBlock(text=result.prompt)])
            ],
            state_changes=(
                CapabilitiesRestricted(capabilities=result.capabilities),
                CompletionTightened(requirements=result.completion_requirements),
            ),
        )
