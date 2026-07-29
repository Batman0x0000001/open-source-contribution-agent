"""定义跨 Control、Worker 与 Publisher 绑定的执行合同。"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Literal

from pydantic import Field, field_serializer

from osc_agent.bot.domain.artifacts import IssuePlanArtifact
from osc_agent.bot.domain.jobs import utc_now
from osc_agent.contracts import ContractModel, FrozenContractModel


class BotApproval(ContractModel):
    approval_id: str = Field(min_length=36, max_length=36)
    job_id: str = Field(min_length=36, max_length=36)
    kind: Literal["implementation"] = "implementation"
    actor_id: int = Field(gt=0)
    actor_login: str = Field(min_length=1)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_hash: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    base_sha: str = Field(default="0" * 40, pattern=r"^[0-9a-f]{40}$")
    status: Literal["approved", "cancelled", "expired"] = "approved"
    expires_at: str
    created_at: str = Field(default_factory=utc_now)


class ExecutionContract(FrozenContractModel):
    contract_version: Literal[1] = 1
    repository_id: int = Field(gt=0)
    repository_full_name: str
    installation_id: int = Field(gt=0)
    base_branch: str
    base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    issue_number: int = Field(gt=0)
    issue_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Literal["anthropic"] = "anthropic"
    model_id: str
    runtime_revision: str = "bot-job-v2"
    plan_profile_revision: str = "bot-plan-v1"
    implementation_profile_revision: str = "bot-implementation-v1"
    planning_skill_name: Literal["issue-planning"] = "issue-planning"
    planning_skill_version: int = 2
    implementation_skill_name: Literal["open-source-contribution"] = "open-source-contribution"
    implementation_skill_version: int = 1
    plan_allowed_tools: frozenset[str]
    implementation_allowed_tools: frozenset[str]
    validation_commands: tuple[str, ...]
    denied_paths: tuple[str, ...]
    max_changed_files: int
    max_patch_bytes: int
    image_id: str
    command_timeout_seconds: int
    container_cpus: float
    container_memory: str
    container_pids: int
    container_network: Literal["none"] = "none"
    pull_request_mode: Literal["draft", "ready"] = "draft"

    @field_serializer("plan_allowed_tools", "implementation_allowed_tools", when_used="json")
    def serialize_allowed_tools(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    @property
    def contract_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def plan_evidence_hash(plan: IssuePlanArtifact) -> str:
    payload = {
        "plan": plan.model_dump(mode="json"),
        "execution_contract_hash": plan.execution_contract_hash,
        "base_sha": plan.base_sha,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_implementation_approval(
    approval: BotApproval | None,
    plan: IssuePlanArtifact | None,
    *,
    now: datetime | None = None,
) -> None:
    if approval is None or approval.status != "approved":
        raise ValueError("implementation approval is missing or inactive")
    if plan is None or plan.status != "ready":
        raise ValueError("approved ready plan is missing")
    try:
        expires_at = datetime.fromisoformat(approval.expires_at)
    except ValueError as exc:
        raise ValueError("implementation approval expiry is invalid") from exc
    current = now or datetime.now(timezone.utc)
    if expires_at.tzinfo is None or expires_at <= current:
        raise ValueError("implementation approval has expired")
    if approval.evidence_hash != plan_evidence_hash(plan):
        raise ValueError("implementation approval does not match the approved plan")
    if approval.execution_contract_hash != plan.execution_contract_hash or approval.base_sha != plan.base_sha:
        raise ValueError("implementation approval does not match the execution contract")
