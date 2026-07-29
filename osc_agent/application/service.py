"""提供产品入口共享的 AgentApplication 和 AgentConversation。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from osc_agent.application.composition import ApplicationGraph, compose_application
from osc_agent.application.models import (
    AgentApplicationConfig,
    AgentInput,
    AgentProfile,
    ResolvedAgentInput,
    SkillInput,
    UserPrompt,
)
from osc_agent.runtime.models import (
    CapabilityScope,
    CompletionRequirements,
    ResumeQueryParams,
    RuntimeEvent,
    RuntimeMessage,
    SessionSnapshot,
    StartQueryParams,
    TextBlock,
)
from osc_agent.skills.models import SkillInvocation


class AgentApplication:
    def __init__(
        self,
        graph: ApplicationGraph,
        *,
        repository_root: Path,
        model: str,
        profile: AgentProfile,
    ) -> None:
        self._graph = graph
        self._repository_root = repository_root.resolve()
        self._model = model
        self._profile = profile

    def open_session(self, session_id: str) -> "AgentConversation":
        if not session_id:
            raise ValueError("session_id is required")
        return AgentConversation(self, session_id)


class AgentConversation:
    def __init__(self, application: AgentApplication, session_id: str) -> None:
        self._application = application
        self.session_id = session_id

    def snapshot(self) -> SessionSnapshot | None:
        return self._application._graph.session_store.load(self.session_id)

    async def submit(self, input: AgentInput | None = None) -> AsyncIterator[RuntimeEvent]:
        if self.snapshot() is None:
            if input is None:
                raise ValueError("new Agent Session requires a UserPrompt or SkillInput")
            async for event in self._start(input):
                yield event
            return
        if isinstance(input, SkillInput):
            raise ValueError("SkillInput can only start a new Agent Session")
        async for event in self._resume(input):
            yield event

    async def _start(self, input: AgentInput) -> AsyncIterator[RuntimeEvent]:
        app = self._application
        resolved = await self._resolve_input(input)
        graph = app._graph
        async for event in graph.runtime.query(
            StartQueryParams(
                session_id=self.session_id,
                model=app._model,
                system_prompt=app._profile.system_prompt + "\n\n" + graph.discovery_prompt,
                messages=list(resolved.messages),
                repository_root=str(app._repository_root),
                capabilities=resolved.capabilities,
                completion_requirements=resolved.completion_requirements,
                config=graph.query_config,
            )
        ):
            yield event

    async def _resume(self, input: UserPrompt | None) -> AsyncIterator[RuntimeEvent]:
        app = self._application
        messages = [] if input is None else [_message(input.text)]
        async for event in app._graph.runtime.query(
            ResumeQueryParams(
                session_id=self.session_id,
                repository_root=str(app._repository_root),
                messages=messages,
                config=app._graph.query_config,
            )
        ):
            yield event

    async def _resolve_input(self, input: AgentInput) -> ResolvedAgentInput:
        app = self._application
        graph = app._graph
        capabilities = (
            CapabilityScope(allowed_tools=app._profile.allowed_tools)
            if app._profile.allowed_tools is not None
            else graph.general_capabilities
        )
        requirements = CompletionRequirements(
            required_evidence=app._profile.required_evidence
        )
        if isinstance(input, UserPrompt):
            return ResolvedAgentInput(
                messages=(_message(input.text),),
                capabilities=capabilities,
                completion_requirements=requirements,
            )
        rendered = await graph.skill_executor.execute(
            SkillInvocation(
                name=input.name,
                arguments=input.arguments,
                session_id=self.session_id,
                working_directory=str(app._repository_root),
                caller_capabilities=capabilities,
                trigger="user",
            )
        )
        if rendered.status != "inline" or rendered.rendered_prompt is None:
            raise ValueError(rendered.error or "Skill did not render inline")
        skill_requirements = rendered.completion_requirements or requirements
        if app._profile.required_evidence:
            skill_requirements = CompletionRequirements(
                required_evidence=(
                    skill_requirements.required_evidence | app._profile.required_evidence
                ),
                waivable_evidence=skill_requirements.waivable_evidence,
            )
        return ResolvedAgentInput(
            messages=(_message(rendered.rendered_prompt),),
            capabilities=rendered.capabilities or capabilities,
            completion_requirements=skill_requirements,
        )


def build_agent_application(config: AgentApplicationConfig) -> AgentApplication:
    graph = compose_application(config)
    return AgentApplication(
        graph,
        repository_root=config.repository_root,
        model=config.settings.model_id or "",
        profile=config.profile,
    )


def _message(text: str) -> RuntimeMessage:
    return RuntimeMessage(role="user", content=[TextBlock(text=text)])
