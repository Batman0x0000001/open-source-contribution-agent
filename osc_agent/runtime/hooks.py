from __future__ import annotations

from typing import Awaitable, Callable, Literal

from pydantic import Field, JsonValue

from osc_agent.runtime.models import ContractModel, ToolResult, ToolUseContext


class PreToolUsePayload(ContractModel):
    tool_name: str = Field(min_length=1)
    input: dict[str, JsonValue]


class PostToolUsePayload(ContractModel):
    tool_name: str = Field(min_length=1)
    input: dict[str, JsonValue]
    result: ToolResult


class HookContinue(ContractModel):
    allowed: Literal[True] = True


class HookBlock(ContractModel):
    allowed: Literal[False] = False
    reason: str = Field(min_length=1)


PreToolHook = Callable[
    [PreToolUsePayload, ToolUseContext],
    Awaitable[HookContinue | HookBlock],
]
PostToolHook = Callable[[PostToolUsePayload, ToolUseContext], Awaitable[None]]


class HookRegistry:
    def __init__(self) -> None:
        self._pre_tool_use: list[PreToolHook] = []
        self._post_tool_use: list[PostToolHook] = []

    def register_pre_tool_use(self, hook: PreToolHook) -> None:
        self._pre_tool_use.append(hook)

    def register_post_tool_use(self, hook: PostToolHook) -> None:
        self._post_tool_use.append(hook)

    async def run_pre_tool_use(
        self,
        payload: PreToolUsePayload,
        context: ToolUseContext,
    ) -> HookContinue | HookBlock:
        for hook in self._pre_tool_use:
            result = await hook(payload, context)
            if not result.allowed:
                return result
        return HookContinue()

    async def run_post_tool_use(
        self,
        payload: PostToolUsePayload,
        context: ToolUseContext,
    ) -> None:
        for hook in self._post_tool_use:
            await hook(payload, context)
