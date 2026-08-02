"""通过共享 Agent Runtime 执行子 Agent。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

from osc_agent.runtime.events import (
    AssistantMessageCompleted,
    Cancelled,
    RunStopped,
)
from osc_agent.runtime.messages import RuntimeMessage, TextBlock
from osc_agent.runtime.query_models import StartQueryParams
from osc_agent.runtime.query import AgentRuntime
from osc_agent.runtime.state import CapabilityScope
from osc_agent.subagents.models import SubagentDefinition, SubagentRequest, SubagentRunResult
from osc_agent.subagents.registry import SubagentRegistration, SubagentRegistry
from osc_agent.subagents.workspace_guard import (
    SubagentWorkspaceError,
    capture_workspace_fingerprint,
    verify_workspace_unchanged,
)


class SubagentRunner:
    """为每个子 Agent 创建独立 Session，并递归复用同一个 Runtime。"""

    def __init__(
        self,
        runtime: AgentRuntime,
        registry: SubagentRegistry,
        *,
        default_model: str,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not default_model:
            raise ValueError("default_model must not be empty")
        self.runtime = runtime
        self.registry = registry
        self.default_model = default_model
        self.session_id_factory = session_id_factory or (lambda: str(uuid4()))
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
        registration = self.registry.get(name)
        if registration is None:
            raise KeyError(f"unknown subagent definition: {name}")
        semaphore = self._semaphores.setdefault(
            name,
            asyncio.Semaphore(registration.max_parallel),
        )
        async with semaphore:
            return await self._run_registered(registration, request)

    async def _run_registered(
        self,
        registration: SubagentRegistration,
        request: SubagentRequest,
    ) -> SubagentRunResult:
        before = None
        if registration.read_only:
            before = await capture_workspace_fingerprint(request.working_directory)
        try:
            result = await self._execute(
                self.session_id_factory(),
                registration.definition,
                request,
            )
        except asyncio.CancelledError as exc:
            if before is not None:
                try:
                    await verify_workspace_unchanged(before, request.working_directory)
                except SubagentWorkspaceError as guard_error:
                    exc.add_note(f"{guard_error.code}: {guard_error}")
            raise
        except Exception:
            if before is not None:
                await verify_workspace_unchanged(before, request.working_directory)
            raise
        if before is None:
            return result
        fingerprint = await verify_workspace_unchanged(
            before,
            request.working_directory,
        )
        return result.model_copy(update={"workspace_fingerprint": fingerprint})

    async def _execute(
        self,
        session_id: str,
        definition: SubagentDefinition,
        request: SubagentRequest,
    ) -> SubagentRunResult:
        messages = [
            RuntimeMessage(role="user", content=[TextBlock(text=request.prompt)])
        ]
        capabilities = self._resolve_capabilities(definition, request)
        output: list[str] = []
        try:
            async for event in self.runtime.query(
                StartQueryParams(
                    session_id=session_id,
                    model=definition.model or self.default_model,
                    system_prompt=definition.system_prompt,
                    messages=messages,
                    workspace_root=request.working_directory,
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
    def _resolve_capabilities(
        definition: SubagentDefinition,
        request: SubagentRequest,
    ) -> CapabilityScope:
        allowed_tools = definition.capabilities.allowed_tools
        if allowed_tools is None:
            raise ValueError("subagent capabilities must explicitly enumerate allowed tools")
        return request.caller_capabilities.intersect(
            CapabilityScope(allowed_tools=allowed_tools)
        )
