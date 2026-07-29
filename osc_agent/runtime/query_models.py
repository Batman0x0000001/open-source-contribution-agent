"""定义单次 Agent Query 的不可变输入契约。"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from osc_agent.completion.models import CompletionRequirements
from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.tool_models import CapabilityScope


class QueryConfig(FrozenContractModel):
    max_rounds: int = Field(default=30, ge=1)
    max_total_tokens: int = Field(default=200_000, ge=1)
    deadline_seconds: int = Field(default=1_800, ge=1)
    max_tool_result_chars: int = Field(default=20_000, ge=1)
    keep_recent_tool_results: int = Field(default=3, ge=0)
    auto_compact_chars: int = Field(default=120_000, ge=1)
    max_reactive_compactions: int = Field(default=1, ge=0)
    max_no_progress_rounds: int = Field(default=3, ge=1)
    max_output_tokens: int = Field(default=8_192, ge=1)


class StartQueryParams(FrozenContractModel):
    mode: Literal["start"] = "start"
    session_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt: str = ""
    messages: list[RuntimeMessage] = Field(default_factory=list)
    workspace_root: str = Field(min_length=1)
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    completion_requirements: CompletionRequirements = Field(default_factory=CompletionRequirements)
    config: QueryConfig = Field(default_factory=QueryConfig)

    @model_validator(mode="after")
    def start_requires_messages(self) -> "StartQueryParams":
        if not self.messages:
            raise ValueError("new query requires at least one message")
        return self


class ResumeQueryParams(FrozenContractModel):
    mode: Literal["resume"] = "resume"
    session_id: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    messages: list[RuntimeMessage] = Field(default_factory=list)
    config: QueryConfig = Field(default_factory=QueryConfig)


QueryParams: TypeAlias = Annotated[
    StartQueryParams | ResumeQueryParams,
    Field(discriminator="mode"),
]
