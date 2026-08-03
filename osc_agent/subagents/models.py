"""定义子 Agent 的静态配置、单次请求和运行结果。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.query_models import QueryConfig
from osc_agent.runtime.state import CapabilityScope


class SubagentDefinition(FrozenContractModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    model: str | None = None
    capabilities: CapabilityScope
    config: QueryConfig = Field(default_factory=QueryConfig)


class SubagentRequest(FrozenContractModel):
    prompt: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    caller_capabilities: CapabilityScope = Field(default_factory=CapabilityScope)


class SubagentRunResult(FrozenContractModel):
    session_id: str = Field(min_length=1)
    status: Literal["completed", "failed", "cancelled"]
    output: str = ""
    error: str | None = None
    workspace_fingerprint: str | None = None
