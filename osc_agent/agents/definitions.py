from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from osc_agent.runtime.models import (
    CapabilityScope,
    FrozenContractModel,
    QueryConfig,
    RuntimeMessage,
)


class AgentDefinition(FrozenContractModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    model: str | None = None
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    config: QueryConfig = Field(default_factory=QueryConfig)
    context_policy: Literal["minimal", "fork"] = "minimal"


class AgentInvocation(FrozenContractModel):
    agent_name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    mode: Literal["inline", "fork"] = "inline"
    parent_session_id: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    caller_capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    parent_messages: list[RuntimeMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def fork_requires_parent_context(self) -> "AgentInvocation":
        if self.mode == "fork" and not self.parent_messages:
            raise ValueError("fork mode requires parent_messages")
        return self


class AgentRunResult(FrozenContractModel):
    session_id: str = Field(min_length=1)
    status: Literal["completed", "failed", "cancelled"]
    output: str = ""
    error: str | None = None
