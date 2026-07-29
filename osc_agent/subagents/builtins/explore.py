"""构建用于只读代码探索的 Explore Agent 注册信息。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.runtime.query_models import QueryConfig
from osc_agent.runtime.state import CapabilityScope
from osc_agent.subagents.models import SubagentDefinition
from osc_agent.subagents.registry import SubagentContract, SubagentRegistration
from osc_agent.workspaces.path_policy import normalize_repo_relative_path, safe_repo_path


EXPLORE_TOOLS = frozenset(
    {
        "read_file",
        "glob",
        "grep",
        "git_status",
        "git_diff",
        "git_log",
    }
)


class ExploreAgentArguments(ContractModel):
    scope_paths: list[str] = Field(default_factory=lambda: ["."], min_length=1, max_length=8)
    context_summary: str | None = Field(default=None, max_length=8_000)

    @field_validator("scope_paths")
    @classmethod
    def validate_scope_paths(cls, values: list[str], info: ValidationInfo) -> list[str]:
        normalized: list[str] = []
        root = _repository_root(info)
        for value in values:
            path = "." if value == "." else normalize_repo_relative_path(value, field_name="scope_paths")
            if root is not None:
                safe_repo_path(root, path)
            if path not in normalized:
                normalized.append(path)
        return normalized


class ExploreEvidence(FrozenContractModel):
    path: str = Field(min_length=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    observation: str = Field(min_length=1, max_length=2_000)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str, info: ValidationInfo) -> str:
        normalized = normalize_repo_relative_path(value)
        root = _repository_root(info)
        if root is not None and not safe_repo_path(root, normalized).is_file():
            raise ValueError(f"evidence file does not exist: {normalized}")
        return normalized

    @model_validator(mode="after")
    def validate_line_range(self, info: ValidationInfo) -> "ExploreEvidence":
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end requires line_start")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")
        root = _repository_root(info)
        if root is not None and self.line_start is not None:
            with safe_repo_path(root, self.path).open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as stream:
                line_count = sum(1 for _line in stream)
            final_line = self.line_end or self.line_start
            if final_line > line_count:
                raise ValueError(
                    f"evidence line {final_line} exceeds {self.path} line count {line_count}"
                )
        return self


class ExploreFinding(FrozenContractModel):
    claim: str = Field(min_length=1, max_length=2_000)
    confidence: Literal["high", "medium", "low"]
    evidence: list[ExploreEvidence] = Field(min_length=1, max_length=5)


class ExploreReport(FrozenContractModel):
    summary: str = Field(min_length=1, max_length=8_000)
    findings: list[ExploreFinding] = Field(default_factory=list, max_length=20)
    relevant_files: list[str] = Field(default_factory=list, max_length=30)
    likely_change_locations: list[str] = Field(default_factory=list, max_length=20)
    recommended_tests: list[str] = Field(default_factory=list, max_length=20)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("relevant_files", "likely_change_locations")
    @classmethod
    def validate_repository_paths(cls, values: list[str], info: ValidationInfo) -> list[str]:
        root = _repository_root(info)
        normalized: list[str] = []
        for value in values:
            path = normalize_repo_relative_path(value)
            if root is not None:
                safe_repo_path(root, path)
            if path not in normalized:
                normalized.append(path)
        return normalized

    @field_validator("recommended_tests", "unresolved_questions")
    @classmethod
    def validate_bounded_text(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 2_000 for value in values):
            raise ValueError("report list entries must contain 1 to 2000 characters")
        return values

    @model_validator(mode="after")
    def require_finding_or_question(self) -> "ExploreReport":
        if not self.findings and not self.unresolved_questions:
            raise ValueError("explore report requires a finding or an unresolved question")
        return self


def build_explore_subagent(*, model: str, config: QueryConfig) -> SubagentRegistration:
    definition = SubagentDefinition(
        name="explore",
        description="Read-only repository exploration with evidence-backed findings",
        system_prompt=(
            "Explore the repository to answer one bounded question. Use only repository evidence. "
            "Do not propose edits as completed work. Return only JSON matching the supplied output schema."
        ),
        model=model,
        capabilities=CapabilityScope(allowed_tools=EXPLORE_TOOLS),
        config=config,
        context_policy="minimal",
    )
    return SubagentRegistration(
        definition=definition,
        input_model=ExploreAgentArguments,
        output_model=ExploreReport,
        prompt_builder=_build_explore_prompt,
        read_only=True,
        concurrency_safe=True,
        max_parallel=2,
    )


def _build_explore_prompt(arguments: SubagentContract) -> str:
    parsed = ExploreAgentArguments.model_validate(arguments)
    return (
        "Scope paths:\n"
        + "\n".join(f"- {path}" for path in parsed.scope_paths)
        + "\n\nContext summary:\n"
        + (parsed.context_summary or "(none)")
        + "\n\nReturn JSON matching this schema:\n"
        + json.dumps(ExploreReport.model_json_schema(mode="validation"), ensure_ascii=False)
    )


def _repository_root(info: ValidationInfo) -> Path | None:
    context = info.context
    if not isinstance(context, dict):
        return None
    value = context.get("repository_root")
    return Path(value).resolve() if isinstance(value, (str, Path)) else None
