"""定义产品入口共享的 Agent 应用、会话和单轮输入。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, JsonValue

from osc_agent.agents.registry import AgentRegistration
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.runtime.hooks import PreToolHook, StopHook
from osc_agent.runtime.models import (
    ApprovalResponse,
    Ask,
    CapabilityScope,
    CompletionRequirements,
    FrozenContractModel,
    RuntimeMessage,
)
from osc_agent.runtime.permissions import PermissionPolicy
from osc_agent.runtime.session_store import SessionStore
from osc_agent.runtime.tool import Tool
from osc_agent.tools.process_runner import ProcessRunner


class AgentProfile(FrozenContractModel):
    profile_id: str = Field(min_length=1)
    system_prompt: str
    allowed_tools: frozenset[str] | None = None
    required_evidence: frozenset[str] = frozenset()


class UserPrompt(FrozenContractModel):
    text: str = Field(min_length=1)


class SkillInput(FrozenContractModel):
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


AgentInput = UserPrompt | SkillInput
ApprovalHandler = Callable[[Ask], Awaitable[ApprovalResponse]]
QuestionHandler = Callable[[list[dict[str, JsonValue]]], Awaitable[dict[str, str]]]


@dataclass(frozen=True)
class AgentApplicationConfig:
    """构建一次产品 Agent 所需的稳定配置和可替换依赖。"""

    settings: Settings
    repository_root: Path
    profile: AgentProfile
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


@dataclass(frozen=True)
class ResolvedAgentInput:
    messages: tuple[RuntimeMessage, ...]
    capabilities: CapabilityScope
    completion_requirements: CompletionRequirements
