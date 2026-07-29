"""定义声明式 Skill 的清单、请求、准备结果和诊断。"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, JsonValue

from osc_agent.completion.models import CompletionRequirements
from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.tool_models import CapabilityScope


SkillSource: TypeAlias = Literal["builtin", "user", "project"]
SkillTrigger: TypeAlias = Literal["user", "model", "product"]


class SkillManifest(FrozenContractModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    when_to_use: str = Field(min_length=1)
    allowed_tools: frozenset[str] = Field(default_factory=frozenset)
    product_tools: frozenset[str] = Field(default_factory=frozenset)
    user_invocable: bool = True
    disable_model_invocation: bool = False
    resources: tuple[str, ...] = ()
    completion: CompletionRequirements = Field(default_factory=CompletionRequirements)


class SkillDescriptor(FrozenContractModel):
    manifest: SkillManifest
    path: str = Field(min_length=1)
    root: str = Field(min_length=1)
    source: SkillSource


class SkillDiagnostic(FrozenContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    path: str = Field(min_length=1)
    source: SkillSource
    skill_name: str | None = None


class SkillDiscovery(FrozenContractModel):
    descriptors: tuple[SkillDescriptor, ...] = ()
    diagnostics: tuple[SkillDiagnostic, ...] = ()


class SkillRequest(FrozenContractModel):
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    trigger: SkillTrigger
    caller_capabilities: CapabilityScope = Field(default_factory=CapabilityScope)


class PreparedSkill(FrozenContractModel):
    name: str = Field(min_length=1)
    prompt: str
    capabilities: CapabilityScope
    completion_requirements: CompletionRequirements = Field(
        default_factory=CompletionRequirements
    )


class SkillPreparationFailure(FrozenContractModel):
    name: str = Field(min_length=1)
    error: str = Field(min_length=1)


SkillPreparationResult: TypeAlias = PreparedSkill | SkillPreparationFailure
