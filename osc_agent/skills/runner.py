"""运行 Skill 命令并维护相关 Runtime 上下文。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Callable

from pydantic import JsonValue

from osc_agent.runtime.models import CapabilityScope, QueryConfig, RuntimeEvent, RuntimeMessage, StartQueryParams, TextBlock
from osc_agent.runtime.query import AgentRuntime
from osc_agent.skills.executor import SkillExecutor
from osc_agent.skills.models import SkillInvocation


class SkillCommandRunner:
    """把显式 inline Skill 调用转换为同一个 AgentRuntime 的首条用户消息。"""

    def __init__(
        self,
        executor: SkillExecutor,
        runtime: AgentRuntime,
        *,
        model: str,
        config: QueryConfig,
        system_prompt: str,
        discovery_prompt: Callable[[CapabilityScope], str] | None = None,
    ) -> None:
        self.executor = executor
        self.runtime = runtime
        self.model = model
        self.config = config
        self.system_prompt = system_prompt
        self.discovery_prompt = discovery_prompt

    async def run(
        self,
        *,
        name: str,
        arguments: dict[str, JsonValue],
        session_id: str,
        working_directory: str,
        capabilities: CapabilityScope | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        scope = capabilities or CapabilityScope()
        result = await self.executor.execute(
            SkillInvocation(
                name=name,
                arguments=arguments,
                session_id=session_id,
                working_directory=working_directory,
                caller_capabilities=scope,
                trigger="user",
            )
        )
        if result.status != "inline":
            raise ValueError(result.error or "explicit skill command requires an inline skill")
        runtime_capabilities = result.capabilities or scope
        system_prompt = self.system_prompt
        if self.discovery_prompt is not None:
            system_prompt += "\n\n" + self.discovery_prompt(runtime_capabilities)
        async for event in self.runtime.query(
            StartQueryParams(
                session_id=session_id,
                model=self.model,
                system_prompt=system_prompt,
                messages=[RuntimeMessage(role="user", content=[TextBlock(text=result.rendered_prompt)])],
                repository_root=working_directory,
                capabilities=runtime_capabilities,
                completion_requirements=result.completion_requirements,
                config=self.config,
            )
        ):
            yield event
