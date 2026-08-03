"""定义 Bot Job 状态与持久化作业模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypeAlias

from pydantic import Field, HttpUrl

from osc_agent.contracts import ContractModel


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
