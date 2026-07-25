from __future__ import annotations

import asyncio

from osc_agent.agents.definitions import AgentDefinition, AgentInvocation, AgentRunResult
from osc_agent.runtime.models import (
    AssistantMessageCompleted,
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
        definitions: list[AgentDefinition],
        *,
        default_model: str,
    ) -> None:
        if not default_model:
            raise ValueError("default_model must not be empty")
        self.runtime = runtime
        self.default_model = default_model
        self._definitions = {definition.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("duplicate agent definition")

    def list_definitions(self) -> list[AgentDefinition]:
        return [self._definitions[name] for name in sorted(self._definitions)]

    async def run(self, invocation: AgentInvocation) -> AgentRunResult:
        definition = self._definition(invocation.agent_name)
        return await self.run_with_definition(definition, invocation)

    async def run_with_definition(
        self,
        definition: AgentDefinition,
        invocation: AgentInvocation,
    ) -> AgentRunResult:
        task_id = self.runtime.dependencies.new_id()
        session_id = self.runtime.dependencies.new_id()
        return await self._execute(task_id, session_id, definition, invocation)

    async def _execute(
        self,
        task_id: str,
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
                        task_id=task_id,
                        session_id=session_id,
                        status="failed",
                        output="\n".join(output),
                        error=event.transition.reason,
                    )
        except asyncio.CancelledError:
            return AgentRunResult(task_id=task_id, session_id=session_id, status="cancelled")
        return AgentRunResult(
            task_id=task_id,
            session_id=session_id,
            status="completed",
            output="\n".join(output),
        )

    def _definition(self, name: str) -> AgentDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown agent definition: {name}") from exc
