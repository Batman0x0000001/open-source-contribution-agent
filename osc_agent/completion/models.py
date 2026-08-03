"""定义可持久化的 Agent 完成要求。"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from osc_agent.contracts import FrozenContractModel


EvidenceKind: TypeAlias = Literal[
    "successful_test",
    "git_change_snapshot",
    "independent_verification",
    "issue_plan",
    "delivery_draft",
]


class CompletionRequirements(FrozenContractModel):
    required_evidence: frozenset[EvidenceKind] = Field(default_factory=frozenset)
    waivable_evidence: frozenset[EvidenceKind] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def waivers_must_be_required(self) -> "CompletionRequirements":
        if not self.waivable_evidence <= self.required_evidence:
            raise ValueError("waivable evidence must also be required")
        return self

    def tighten(self, other: "CompletionRequirements") -> "CompletionRequirements":
        """合并要求时只能收紧，且结果与合并顺序无关。"""

        non_waivable = (
            self.required_evidence - self.waivable_evidence
        ) | (other.required_evidence - other.waivable_evidence)
        return CompletionRequirements(
            required_evidence=self.required_evidence | other.required_evidence,
            waivable_evidence=(
                (self.waivable_evidence | other.waivable_evidence) - non_waivable
            ),
        )
