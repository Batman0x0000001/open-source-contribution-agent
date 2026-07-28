from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from osc_agent.agents.registry import AgentRegistration
from osc_agent.application import (
    ApplicationServices,
    ApprovalHandler,
    QuestionHandler,
    build_application,
)
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.runtime.models import (
    CapabilityScope,
    CompletionRequirements,
    FrozenContractModel,
    ResumeQueryParams,
    RuntimeEvent,
    RuntimeMessage,
    StartQueryParams,
    TextBlock,
)
from osc_agent.skills.models import SkillInvocation
from osc_agent.runtime.hooks import PreToolHook, StopHook
from osc_agent.runtime.permissions import PermissionPolicy
from osc_agent.runtime.session_store import SessionStore
from osc_agent.runtime.tool import Tool
from osc_agent.tools.process import ProcessRunner


class InboundMessage(FrozenContractModel):
    source: Literal["github", "cli"]
    source_id: str
    text: str


class AgentRunSpec(FrozenContractModel):
    profile: Literal["bot_plan", "bot_implementation", "local_debug"]
    session_id: str
    repository_root: str
    skill_name: str | None = None
    skill_arguments: dict[str, JsonValue] = Field(default_factory=dict)
    user_message: str | None = None
    execution_contract_hash: str | None = None
    inbound_message: InboundMessage | None = None


class AgentRunProfile(FrozenContractModel):
    name: Literal["bot_plan", "bot_implementation", "local_debug"]
    system_prompt: str
    allowed_tools: frozenset[str] | None = None
    required_evidence: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RunEnvironment:
    """Runtime composition inputs; product entrypoints never assemble Query parameters."""

    settings: Settings
    repository_root: Path
    approval_handler: ApprovalHandler | None = None
    question_handler: QuestionHandler | None = None
    model_gateway: ModelGateway | None = None
    session_store: SessionStore | None = None
    state_root: Path | None = None
    process_runner: ProcessRunner | None = None
    permission_policy: PermissionPolicy | None = None
    extra_tools: tuple[Tool, ...] = ()
    pre_tool_hooks: tuple[PreToolHook, ...] = ()
    stop_hooks: tuple[StopHook, ...] = ()
    agent_registrations: tuple[AgentRegistration, ...] | None = None


class AgentApplicationService:
    """The sole production path from a product entrypoint into AgentRuntime."""

    def __init__(self, services: ApplicationServices, *, model: str, profile: AgentRunProfile) -> None:
        self.services = services
        self.model = model
        self.profile = profile

    async def run(self, spec: AgentRunSpec) -> AsyncIterator[RuntimeEvent]:
        if spec.profile != self.profile.name:
            raise ValueError("Agent run profile does not match the composed application")
        existing = self.services.session_store.load(spec.session_id)
        messages: list[RuntimeMessage] = []
        if spec.inbound_message is not None:
            messages.append(RuntimeMessage(role="user", content=[TextBlock(text=spec.inbound_message.text)]))
        elif spec.user_message:
            messages.append(RuntimeMessage(role="user", content=[TextBlock(text=spec.user_message)]))
        if existing is not None:
            async for event in self.services.runtime.query(
                ResumeQueryParams(
                    session_id=spec.session_id,
                    messages=messages,
                    repository_root=spec.repository_root,
                    config=self.services.query_config,
                )
            ):
                yield event
            return

        capabilities = (
            CapabilityScope(allowed_tools=self.profile.allowed_tools)
            if self.profile.allowed_tools is not None
            else self.services.general_capabilities
        )
        prompt = spec.user_message or ""
        requirements = CompletionRequirements(required_evidence=self.profile.required_evidence)
        if spec.skill_name is not None:
            rendered = await self.services.skill_executor.execute(
                SkillInvocation(
                    name=spec.skill_name,
                    arguments=spec.skill_arguments,
                    session_id=spec.session_id,
                    working_directory=spec.repository_root,
                    caller_capabilities=capabilities,
                    trigger="internal",
                )
            )
            if rendered.status != "inline" or rendered.rendered_prompt is None:
                raise ValueError(rendered.error or "Skill did not render inline")
            prompt = rendered.rendered_prompt
            capabilities = rendered.capabilities or capabilities
            requirements = rendered.completion_requirements or requirements
            if self.profile.required_evidence:
                requirements = CompletionRequirements(
                    required_evidence=requirements.required_evidence | self.profile.required_evidence,
                    waivable_evidence=requirements.waivable_evidence,
                )
        if not prompt:
            raise ValueError("new Agent Session requires a user message or Skill")
        async for event in self.services.runtime.query(
            StartQueryParams(
                session_id=spec.session_id,
                model=self.model,
                system_prompt=self.profile.system_prompt + "\n\n" + self.services.discovery_prompt,
                messages=[RuntimeMessage(role="user", content=[TextBlock(text=prompt)])],
                repository_root=spec.repository_root,
                capabilities=capabilities,
                completion_requirements=requirements,
                config=self.services.query_config,
            )
        ):
            yield event


def build_agent_application(
    *,
    environment: RunEnvironment,
    profile: AgentRunProfile,
) -> AgentApplicationService:
    services = build_application(
        settings=environment.settings,
        repo_root=environment.repository_root,
        approval_handler=environment.approval_handler,
        question_handler=environment.question_handler,
        model_gateway=environment.model_gateway,
        session_store_override=environment.session_store,
        state_root_override=environment.state_root,
        process_runner=environment.process_runner,
        permission_policy=environment.permission_policy,
        extra_tools=environment.extra_tools,
        pre_tool_hooks=environment.pre_tool_hooks,
        stop_hooks=environment.stop_hooks,
        agent_registrations=environment.agent_registrations,
    )
    return AgentApplicationService(
        services, model=environment.settings.model_id or "", profile=profile
    )
