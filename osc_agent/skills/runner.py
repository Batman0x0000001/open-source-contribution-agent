from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import JsonValue

from osc_agent.runtime.models import CapabilityScope, QueryConfig, RuntimeEvent, RuntimeMessage, StartQueryParams, TextBlock
from osc_agent.runtime.query import AgentRuntime
from osc_agent.skills.executor import SkillExecutor
from osc_agent.skills.models import SkillInvocation


class SkillCommandRunner:
    """把显式 inline Skill 调用转换为同一个 AgentRuntime 的首条用户消息。"""

    def __init__(self, executor: SkillExecutor, runtime: AgentRuntime, *, model: str, config: QueryConfig, system_prompt: str) -> None:
        self.executor = executor
        self.runtime = runtime
        self.model = model
        self.config = config
        self.system_prompt = system_prompt

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
        async for event in self.runtime.query(
            StartQueryParams(
                session_id=session_id,
                model=self.model,
                system_prompt=self.system_prompt,
                messages=[RuntimeMessage(role="user", content=[TextBlock(text=result.rendered_prompt)])],
                repository_root=working_directory,
                capabilities=result.capabilities or scope,
                config=self.config,
            )
        ):
            yield event
