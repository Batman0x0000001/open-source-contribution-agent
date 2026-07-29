"""定义 Bot 作业、执行合同、仓库策略和发布产物模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import PurePosixPath
from typing import Literal, TypeAlias

from pydantic import Field, HttpUrl, field_serializer, field_validator, model_validator

from osc_agent.runtime.models import ContractModel, FrozenContractModel


JobStatus: TypeAlias = Literal[
    "queued_plan",
    "running_plan",
    "blocked_plan",
    "waiting_approval",
    "queued_implementation",
    "running_implementation",
    "ready_to_publish",
    "publishing",
    "completed",
    "retry_wait",
    "stale",
    "dead_letter",
    "cancelled",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RepositoryBotConfig(FrozenContractModel):
    enabled: bool = True
    image: str = Field(
        pattern=r"^sha256:[0-9a-f]{64}$",
        description="Worker 已预构建并固定的不可变 Docker image ID。",
    )
    validation_commands: tuple[str, ...] = Field(min_length=1, max_length=20)
    denied_paths: tuple[str, ...] = (
        ".git/**",
        ".github/workflows/**",
        ".gitmodules",
        ".osc_agent/**",
    )
    max_changed_files: int = Field(default=100, ge=1, le=1_000)
    max_patch_bytes: int = Field(default=2_000_000, ge=1, le=50_000_000)
    command_timeout_seconds: int = Field(default=1_800, ge=1, le=7_200)
    container_cpus: float = Field(default=2.0, gt=0, le=32)
    container_memory: str = Field(default="4g", pattern=r"^[1-9][0-9]*(?:m|g)$")
    container_pids: int = Field(default=256, ge=16, le=4_096)
    pull_request_mode: Literal["draft", "ready"] = "draft"

    @field_validator("validation_commands", "denied_paths", mode="before")
    @classmethod
    def normalize_yaml_sequences(cls, value: object) -> object:
        # YAML represents sequences as lists. Normalize at this explicit config
        # boundary while retaining immutable tuples in the runtime contract.
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_commands_and_paths(self) -> "RepositoryBotConfig":
        if any(not command.strip() for command in self.validation_commands):
            raise ValueError("validation commands must be non-empty")
        from osc_agent.processes.contracts import CommandKind
        from osc_agent.processes.policy import classify_command

        if not any(classify_command(command) == CommandKind.TEST for command in self.validation_commands):
            raise ValueError("validation commands must include at least one recognized test command")
        for pattern in self.denied_paths:
            normalized = pattern.replace("\\", "/")
            if not normalized or PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
                raise ValueError("denied paths must be repository-relative patterns")
        return self


class RepositoryBotCatalog(FrozenContractModel):
    repositories: dict[str, RepositoryBotConfig]

    @model_validator(mode="after")
    def validate_names(self) -> "RepositoryBotCatalog":
        for name in self.repositories:
            parts = name.split("/")
            if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
                raise ValueError(f"invalid GitHub repository name: {name}")
        return self


class BotJob(ContractModel):
    job_id: str = Field(min_length=36, max_length=36)
    repository_id: int = Field(gt=0)
    repository_full_name: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    installation_id: int = Field(gt=0)
    issue_number: int = Field(gt=0)
    issue_url: HttpUrl
    base_branch: str = "main"
    base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    issue_input_hash: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    execution_contract_hash: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    image_id: str = Field(min_length=1, max_length=300)
    status: JobStatus
    plan_session_id: str | None = None
    implementation_session_id: str | None = None
    plan_artifact_id: str | None = None
    approval_id: str | None = None
    plan_workspace_path: str | None = None
    plan_workspace_ready: bool = False
    implementation_workspace_path: str | None = None
    implementation_workspace_ready: bool = False
    last_progress_event: str | None = Field(default=None, max_length=64)
    last_progress_at: str | None = None
    retry_phase: Literal["plan", "implementation", "publish"] | None = None
    plan_attempts: int = Field(default=0, ge=0)
    implementation_attempts: int = Field(default=0, ge=0)
    publish_attempts: int = Field(default=0, ge=0)
    branch: str | None = None
    commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    pull_request_number: int | None = Field(default=None, gt=0)
    pull_request_url: HttpUrl | None = None
    lease_owner: str | None = None
    lease_until: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    version: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


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
        for collection in (
            self.assumptions,
            self.unresolved_questions,
            self.recommended_tests,
        ):
            if any(not value.strip() or len(value) > 1_000 for value in collection):
                raise ValueError("plan artifact list entries must be non-empty and bounded")
        for value in self.likely_change_paths:
            path = PurePosixPath(value.replace("\\", "/"))
            if (
                not value.strip()
                or len(value) > 500
                or path.is_absolute()
                or ".." in path.parts
            ):
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


class OutboxEvent(ContractModel):
    event_id: str = Field(min_length=1)
    job_id: str = Field(min_length=36, max_length=36)
    kind: Literal["prepare", "issue_comment", "publish"]
    idempotency_key: str = Field(min_length=1, max_length=200)
    payload: dict[str, object]
    status: Literal["pending", "processing", "completed", "failed", "dead_letter"] = "pending"
    attempts: int = Field(default=0, ge=0)
    next_attempt_at: str = Field(default_factory=utc_now)
    last_error: str | None = None


class WebhookCommand(FrozenContractModel):
    action: Literal["plan", "implement", "reply", "status", "retry", "cancel"]
    job_id: str | None = None
    message: str | None = None


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
        # 契约跨 Control/Worker 进程持久化，集合必须排序后再参与哈希。
        return sorted(value)

    @property
    def contract_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def plan_evidence_hash(plan: IssuePlanArtifact) -> str:
    """Return the stable digest bound to an implementation approval."""

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
