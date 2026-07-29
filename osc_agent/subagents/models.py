"""定义子 Agent 的静态配置、单次请求和运行结果。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from osc_agent.runtime.models import (
    CapabilityScope,
    FrozenContractModel,
    QueryConfig,
    RuntimeMessage,
)


class SubagentDefinition(FrozenContractModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    model: str | None = None
    capabilities: CapabilityScope
    config: QueryConfig = Field(default_factory=QueryConfig)
    context_policy: Literal["minimal", "fork"] = "minimal"


class SubagentRequest(FrozenContractModel):
    prompt: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    caller_capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    parent_messages: tuple[RuntimeMessage, ...] = ()


class SubagentRunResult(FrozenContractModel):
    session_id: str = Field(min_length=1)
    status: Literal["completed", "failed", "cancelled"]
    output: str = ""
    error: str | None = None
