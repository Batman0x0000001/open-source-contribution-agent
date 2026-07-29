"""定义 Plan 与 Implementation 产生的严格交付 Artifact。"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from osc_agent.contracts import ContractModel


class IssuePlanArtifact(ContractModel):
    evidence_type: Literal["issue_plan"] = "issue_plan"
    status: Literal["ready", "blocked"]
    base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_contract_hash: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    summary: str = Field(min_length=1, max_length=4_000)
    plan_markdown: str = Field(min_length=1, max_length=50_000)
    assumptions: list[str] = Field(default_factory=list, max_length=30)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
    recommended_tests: list[str] = Field(default_factory=list, max_length=30)
    likely_change_paths: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def status_matches_questions(self) -> "IssuePlanArtifact":
        if self.status == "ready" and self.unresolved_questions:
            raise ValueError("ready plan cannot contain unresolved questions")
        if self.status == "blocked" and not self.unresolved_questions:
            raise ValueError("blocked plan must contain unresolved questions")
        for collection in (self.assumptions, self.unresolved_questions, self.recommended_tests):
            if any(not value.strip() or len(value) > 1_000 for value in collection):
                raise ValueError("plan artifact list entries must be non-empty and bounded")
        for value in self.likely_change_paths:
            path = PurePosixPath(value.replace("\\", "/"))
            if not value.strip() or len(value) > 500 or path.is_absolute() or ".." in path.parts:
                raise ValueError("likely change paths must be bounded repository-relative paths")
        return self


class DeliveryDraft(ContractModel):
    evidence_type: Literal["delivery_draft"] = "delivery_draft"
    title: str = Field(min_length=1, max_length=72)
    body: str = Field(min_length=1, max_length=50_000)
    commit_message: str = Field(min_length=1, max_length=72)
    issue_number: int = Field(gt=0)
    base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_contract_hash: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_summary: str = Field(min_length=1, max_length=4_000)
    verification_summary: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def single_line_headers(self) -> "DeliveryDraft":
        if any(character in self.title for character in "\r\n"):
            raise ValueError("delivery title must be a single line")
        if any(character in self.commit_message for character in "\r\n"):
            raise ValueError("commit message must be a single line")
        return self
