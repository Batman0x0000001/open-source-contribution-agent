from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import JsonValue

from osc_agent.runtime.models import (
    Allow,
    ContractModel,
    PermissionDecision,
    ToolResult,
    ToolUseContext,
    ValidationResult,
    ValidationSuccess,
)


InputT = TypeVar("InputT", bound=ContractModel)
OutputT = TypeVar("OutputT", bound=ContractModel)


@runtime_checkable
class Tool(Protocol[InputT, OutputT]):
    name: str
    description: str
    input_model: type[InputT]
    output_model: type[OutputT]

    def is_enabled(self) -> bool: ...

    def is_read_only(self, input: InputT) -> bool: ...

    def is_concurrency_safe(self, input: InputT) -> bool: ...

    def is_destructive(self, input: InputT) -> bool: ...

    def permission_risk(self, input: InputT) -> str: ...

    def permission_preview(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> dict[str, JsonValue]: ...

    async def validate_input(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> ValidationResult: ...

    async def check_permissions(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> PermissionDecision: ...

    async def call(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> ToolResult: ...


class BaseTool(Generic[InputT, OutputT]):
    """提供 Claude Code buildTool 风格的保守默认值。"""

    name: str
    description: str = "Execute this Tool using its validated input and output contracts."
    input_model: type[InputT]
    output_model: type[OutputT]

    def is_enabled(self) -> bool:
        return True

    def is_read_only(self, input: InputT) -> bool:
        return False

    def is_concurrency_safe(self, input: InputT) -> bool:
        return False

    def is_destructive(self, input: InputT) -> bool:
        return False

    def permission_risk(self, input: InputT) -> str:
        return "destructive"

    def permission_preview(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> dict[str, JsonValue]:
        return _bounded_preview(input.model_dump(mode="json"))

    async def validate_input(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> ValidationResult:
        return ValidationSuccess()

    async def check_permissions(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> PermissionDecision:
        updated_input: dict[str, JsonValue] = input.model_dump(mode="json")
        return Allow(updated_input=updated_input)

    @abstractmethod
    async def call(
        self,
        input: InputT,
        context: ToolUseContext,
    ) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: list[Tool[ContractModel, ContractModel]] | None = None) -> None:
        self._tools: dict[str, Tool[ContractModel, ContractModel]] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool[ContractModel, ContractModel]) -> None:
        name = tool.name.strip()
        if not name:
            raise ValueError("tool name must not be empty")
        if name in self._tools:
            raise ValueError(f"duplicate tool: {name}")
        if not tool.description.strip():
            raise ValueError(f"tool {name} description must not be empty")
        self._tools[name] = tool

    def get(self, name: str) -> Tool[ContractModel, ContractModel] | None:
        tool = self._tools.get(name)
        return tool if tool is not None and tool.is_enabled() else None

    def names(self) -> list[str]:
        return sorted(name for name, tool in self._tools.items() if tool.is_enabled())

    def available(self, context: ToolUseContext) -> list[Tool[ContractModel, ContractModel]]:
        return [
            tool
            for tool in self._tools.values()
            if tool.is_enabled() and context.capabilities.permits_tool(tool.name)
        ]

    def schemas(self, context: ToolUseContext) -> list[dict[str, JsonValue]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(mode="validation"),
            }
            for tool in self.available(context)
        ]


def _bounded_preview(values: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """默认预览必须有界，并避免把常见凭据字段回显到审批界面。"""

    preview: dict[str, JsonValue] = {}
    for key, value in values.items():
        lowered = key.casefold()
        if any(marker in lowered for marker in ("token", "secret", "password", "api_key")):
            preview[key] = "[REDACTED]"
        elif isinstance(value, str) and len(value) > 500:
            preview[key] = value[:500] + f"… [{len(value)} chars]"
        else:
            preview[key] = value
    return preview
