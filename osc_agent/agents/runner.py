from __future__ import annotations

import asyncio

from osc_agent.agents.definitions import AgentDefinition, AgentInvocation, AgentRunResult
from osc_agent.agents.registry import AgentRegistry
from osc_agent.runtime.models import (
    AssistantMessageCompleted,
    Cancelled,
    StartQueryParams,
    RunStopped,
    RuntimeMessage,
    TextBlock,
)
from osc_agent.runtime.query import AgentRuntime


class AgentRunner:
    """所有子 Agent 都递归调用同一个 AgentRuntime 实例。"""

    def __init__(
        self,
        runtime: AgentRuntime,
        registry: AgentRegistry,
        *,
        default_model: str,
    ) -> None:
        if not default_model:
            raise ValueError("default_model must not be empty")
        self.runtime = runtime
        self.registry = registry
        self.default_model = default_model

    def list_definitions(self) -> list[AgentDefinition]:
        return [registration.definition for registration in self.registry.list()]

    async def run(self, invocation: AgentInvocation) -> AgentRunResult:
        definition = self._definition(invocation.agent_name)
        return await self.run_with_definition(definition, invocation)

    async def run_with_definition(
        self,
        definition: AgentDefinition,
        invocation: AgentInvocation,
    ) -> AgentRunResult:
        session_id = self.runtime.dependencies.new_id()
        return await self._execute(session_id, definition, invocation)

    async def _execute(
        self,
        session_id: str,
        definition: AgentDefinition,
        invocation: AgentInvocation,
    ) -> AgentRunResult:
        messages = (
            [message.model_copy(deep=True) for message in invocation.parent_messages]
            if invocation.mode == "fork"
            else []
        )
        messages.append(RuntimeMessage(role="user", content=[TextBlock(text=invocation.prompt)]))
        capabilities = invocation.caller_capabilities.intersect(definition.capabilities)
        output: list[str] = []
        try:
            async for event in self.runtime.query(
                StartQueryParams(
                    session_id=session_id,
                    model=definition.model or self.default_model,
                    system_prompt=definition.system_prompt,
                    messages=messages,
                    repository_root=invocation.working_directory,
                    capabilities=capabilities,
                    config=definition.config,
                )
            ):
                if isinstance(event, AssistantMessageCompleted):
                    output = [block.text for block in event.message.content if isinstance(block, TextBlock)]
                elif isinstance(event, RunStopped):
                    return AgentRunResult(
                        session_id=session_id,
                        status="cancelled" if isinstance(event.transition, Cancelled) else "failed",
                        output="\n".join(output),
                        error=event.transition.reason,
                    )
        except asyncio.CancelledError:
            raise
        return AgentRunResult(
            session_id=session_id,
            status="completed",
            output="\n".join(output),
        )

    def _definition(self, name: str) -> AgentDefinition:
        registration = self.registry.get(name)
        if registration is None:
            raise KeyError(f"unknown agent definition: {name}")
        return registration.definition
