"""定义工具调用的权限决策接口和默认策略。"""

from __future__ import annotations

from typing import Protocol

from pydantic import JsonValue

from osc_agent.contracts import ContractModel
from osc_agent.runtime.state import ToolContext
from osc_agent.runtime.tool_models import Allow, Ask, Deny, PermissionDecision
from osc_agent.runtime.tool import Tool


class PermissionPolicy(Protocol):
    async def decide(
        self,
        tool: Tool[ContractModel, ContractModel],
        input: ContractModel,
        context: ToolContext,
    ) -> PermissionDecision: ...


class DefaultPermissionPolicy:
    """最小通用策略；Capability 硬门禁由 ToolExecutor 统一执行。"""

    async def decide(
        self,
        tool: Tool[ContractModel, ContractModel],
        input: ContractModel,
        context: ToolContext,
    ) -> PermissionDecision:
        if (
            context.permissions.mode == "plan"
            and not tool.is_read_only(input)
            and tool.name != "write_plan"
        ):
            return Deny(reason=f"tool {tool.name} is not allowed in plan mode")
        if tool.is_destructive(input):
            return Ask(
                tool_name=tool.name,
                prompt=f"Allow {tool.permission_risk(input)} tool call {tool.name}?",
                working_directory=context.workspace.working_directory,
                risk=tool.permission_risk(input),
                preview=tool.permission_preview(input, context),
            )
        updated_input: dict[str, JsonValue] = input.model_dump(mode="json")
        return Allow(updated_input=updated_input)
