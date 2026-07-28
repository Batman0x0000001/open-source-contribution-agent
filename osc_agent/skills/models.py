"""定义 Skill 清单、调用参数和资源模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from osc_agent.runtime.models import (
    CapabilityScope,
    CompletionRequirements,
    FrozenContractModel,
    RuntimeMessage,
)


class SkillManifest(FrozenContractModel):
    name: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    description: str = Field(min_length=1)
    when_to_use: str = Field(min_length=1)
    allowed_tools: frozenset[str] = Field(default_factory=frozenset)
    context: Literal["inline", "fork"] = "inline"
    model: str | None = None
    user_invocable: bool = True
    disable_model_invocation: bool = False
    resources: tuple[str, ...] = ()
    completion: CompletionRequirements = Field(default_factory=CompletionRequirements)
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]


class SkillDescriptor(FrozenContractModel):
    manifest: SkillManifest
    path: str = Field(min_length=1)
    root: str = Field(min_length=1)
    source: Literal["builtin", "user", "project"]


class SkillInvocation(FrozenContractModel):
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    session_id: str = Field(min_length=1)
    working_directory: str = Field(min_length=1)
    caller_capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    parent_messages: list[RuntimeMessage] = Field(default_factory=list)
    trigger: Literal["user", "model", "internal"] = "internal"


class SkillResult(FrozenContractModel):
    name: str = Field(min_length=1)
    status: Literal["inline", "completed", "failed"]
    rendered_prompt: str = ""
    output: JsonValue = None
    error: str | None = None
    capabilities: CapabilityScope | None = None
    completion_requirements: CompletionRequirements = Field(
        default_factory=CompletionRequirements
    )
