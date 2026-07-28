from __future__ import annotations

import json
from typing import Literal

from pydantic import Field, field_validator, model_validator

from osc_agent.agents.definitions import AgentDefinition
from osc_agent.agents.registry import AgentContract, AgentRegistration
from osc_agent.runtime.models import CapabilityScope, ContractModel, FrozenContractModel, QueryConfig


VERIFY_TOOLS = frozenset(
    {
        "read_file",
        "glob",
        "grep",
        "git_status",
        "git_diff",
        "git_log",
        "bash",
    }
)


class VerifyAgentArguments(ContractModel):
    original_goal: str = Field(min_length=1, max_length=8_000)
    implementation_summary: str = Field(min_length=1, max_length=8_000)
    focus_areas: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("focus_areas")
    @classmethod
    def validate_focus_areas(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 1_000 for value in values):
            raise ValueError("focus areas must contain 1 to 1000 characters")
        return values


class VerificationCheck(FrozenContractModel):
    name: str = Field(min_length=1, max_length=500)
    command: str = Field(min_length=1, max_length=4_000)
    exit_code: int | None = None
    output_excerpt: str = Field(max_length=8_000)
    result: Literal["pass", "fail", "blocked"]
    adversarial: bool = False

    @model_validator(mode="after")
    def execution_matches_result(self) -> "VerificationCheck":
        if self.result == "blocked" and self.exit_code is not None:
            raise ValueError("blocked checks must not contain an exit code")
        if self.result != "blocked" and self.exit_code is None:
            raise ValueError("executed checks require an exit code")
        return self


class VerificationReport(FrozenContractModel):
    evidence_type: Literal["independent_verification"]
    verdict: Literal["PASS", "FAIL", "PARTIAL"]
    summary: str = Field(min_length=1, max_length=8_000)
    checks: list[VerificationCheck] = Field(min_length=1, max_length=20)
    risks: list[str] = Field(default_factory=list, max_length=10)
    unverified: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("risks", "unverified")
    @classmethod
    def validate_bounded_text(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 2_000 for value in values):
            raise ValueError("report list entries must contain 1 to 2000 characters")
        return values

    @model_validator(mode="after")
    def verdict_matches_checks(self) -> "VerificationReport":
        failed = [check for check in self.checks if check.result == "fail"]
        blocked = [check for check in self.checks if check.result == "blocked"]
        if failed and self.verdict != "FAIL":
            raise ValueError("reports containing failed checks must use verdict FAIL")
        if self.verdict == "FAIL" and not failed:
            raise ValueError("FAIL requires at least one failed check")
        if self.verdict == "PASS":
            if blocked or self.unverified:
                raise ValueError("PASS cannot contain blocked or unverified checks")
            if not any(
                check.adversarial and check.exit_code is not None
                for check in self.checks
            ):
                raise ValueError("PASS requires an executed adversarial check")
        if self.verdict == "PARTIAL":
            if failed:
                raise ValueError("PARTIAL cannot contain failed checks")
            if not (blocked or self.unverified):
                raise ValueError("PARTIAL requires a blocked check or unverified item")
            if not self.risks:
                raise ValueError("PARTIAL requires explicit risks")
        return self


def build_verify_registration(*, model: str, config: QueryConfig) -> AgentRegistration:
    definition = AgentDefinition(
        name="verify",
        description="Independently run checks and adversarial probes without modifying the repository",
        system_prompt=(
            "You are an independent verification specialist. Try to break the implementation rather "
            "than confirming it by inspection. Do not modify the repository or install dependencies. "
            "Run applicable build, test, lint, typecheck, CLI, or focused reproduction commands and "
            "include at least one adversarial probe before PASS. Return only JSON matching the supplied "
            "output schema. Use FAIL for observed failures and PARTIAL only for environmental blockers."
        ),
        model=model,
        capabilities=CapabilityScope(allowed_tools=VERIFY_TOOLS),
        config=config,
        context_policy="minimal",
    )
    return AgentRegistration(
        definition=definition,
        input_model=VerifyAgentArguments,
        output_model=VerificationReport,
        prompt_builder=_build_verify_prompt,
        read_only=True,
        concurrency_safe=False,
        max_parallel=1,
    )


def _build_verify_prompt(arguments: AgentContract) -> str:
    parsed = VerifyAgentArguments.model_validate(arguments)
    focus = "\n".join(f"- {item}" for item in parsed.focus_areas) or "- (none)"
    return (
        f"Original goal:\n{parsed.original_goal}\n\n"
        f"Implementation summary:\n{parsed.implementation_summary}\n\n"
        f"Focus areas:\n{focus}\n\n"
        "Inspect the actual Git diff and repository instructions. Execute checks yourself; do not trust "
        "the parent summary as evidence. Every check must include its exact command and observed output. "
        "Return only one JSON object matching this schema. Do not wrap the JSON in Markdown or add prose. "
        "Every pass or fail check requires an integer exit_code; blocked checks require exit_code null. "
        "PASS requires an empty unverified list and at least one executed adversarial check. If anything "
        "remains unverified, return PARTIAL with explicit risks and at least one blocked check.\n"
        + json.dumps(VerificationReport.model_json_schema(mode="validation"), ensure_ascii=False)
    )
