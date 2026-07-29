"""通过共享 Agent Runtime 执行子 Agent。"""

from __future__ import annotations

import asyncio

from osc_agent.runtime.models import (
    AssistantMessageCompleted,
    Cancelled,
    CapabilityScope,
    RunStopped,
    RuntimeMessage,
    StartQueryParams,
    TextBlock,
)
from osc_agent.runtime.query import AgentRuntime
from osc_agent.subagents.models import SubagentDefinition, SubagentRequest, SubagentRunResult
from osc_agent.subagents.registry import SubagentRegistry


class SubagentRunner:
    """为每个子 Agent 创建独立 Session，并递归复用同一个 Runtime。"""

    def __init__(
        self,
        runtime: AgentRuntime,
        registry: SubagentRegistry,
        *,
        default_model: str,
    ) -> None:
        if not default_model:
            raise ValueError("default_model must not be empty")
        self.runtime = runtime
        self.registry = registry
        self.default_model = default_model

    async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
        registration = self.registry.get(name)
        if registration is None:
            raise KeyError(f"unknown subagent definition: {name}")
        return await self.run_definition(registration.definition, request)

    async def run_definition(
        self,
        definition: SubagentDefinition,
        request: SubagentRequest,
    ) -> SubagentRunResult:
        session_id = self.runtime.dependencies.new_id()
        return await self._execute(session_id, definition, request)

    async def _execute(
        self,
        session_id: str,
        definition: SubagentDefinition,
        request: SubagentRequest,
    ) -> SubagentRunResult:
        messages = self._resolve_messages(definition, request)
        capabilities = self._resolve_capabilities(definition, request)
        output: list[str] = []
        try:
            async for event in self.runtime.query(
                StartQueryParams(
                    session_id=session_id,
                    model=definition.model or self.default_model,
                    system_prompt=definition.system_prompt,
                    messages=messages,
                    repository_root=request.working_directory,
                    capabilities=capabilities,
                    config=definition.config,
                )
            ):
                if isinstance(event, AssistantMessageCompleted):
                    output = [block.text for block in event.message.content if isinstance(block, TextBlock)]
                elif isinstance(event, RunStopped):
                    return SubagentRunResult(
                        session_id=session_id,
                        status="cancelled" if isinstance(event.transition, Cancelled) else "failed",
                        output="\n".join(output),
                        error=event.transition.reason,
                    )
        except asyncio.CancelledError:
            raise
        return SubagentRunResult(
            session_id=session_id,
            status="completed",
            output="\n".join(output),
        )

    @staticmethod
    def _resolve_messages(
        definition: SubagentDefinition,
        request: SubagentRequest,
    ) -> list[RuntimeMessage]:
        if definition.context_policy == "fork":
            if not request.parent_messages:
                raise ValueError("fork subagent requires parent_messages")
            messages = [message.model_copy(deep=True) for message in request.parent_messages]
        else:
            messages = []
        messages.append(RuntimeMessage(role="user", content=[TextBlock(text=request.prompt)]))
        return messages

    @staticmethod
    def _resolve_capabilities(
        definition: SubagentDefinition,
        request: SubagentRequest,
    ) -> CapabilityScope:
        allowed_tools = definition.capabilities.allowed_tools
        if allowed_tools is None:
            raise ValueError("subagent capabilities must explicitly enumerate allowed tools")
        bounded_definition = CapabilityScope(allowed_tools=allowed_tools - {"agent"})
        return request.caller_capabilities.intersect(bounded_definition)
