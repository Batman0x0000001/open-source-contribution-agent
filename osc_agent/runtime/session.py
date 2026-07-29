"""定义可持久化的 Session V4 数据模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from osc_agent.completion.models import CompletionRequirements
from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.tool_models import CapabilityScope, PermissionGrant
from osc_agent.workspaces.models import (
    FileObservation,
    RepositoryInstructionState,
    WorktreeSession,
)


class SessionMetadata(FrozenContractModel):
    schema_version: Literal[4]
    session_id: str = Field(min_length=1)
    # V4 历史字段名；其值是创建 Query 时的 workspace root。
    repository_root: str = Field(min_length=1)
    initial_working_directory: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt: str = ""
    capabilities: CapabilityScope = Field(default_factory=CapabilityScope)
    completion_requirements: CompletionRequirements = Field(default_factory=CompletionRequirements)


class SessionRuntimeState(ContractModel):
    permission_mode: Literal["default", "plan"] = "default"
    plan_path: str | None = None
    worktree: WorktreeSession | None = None
    capabilities: CapabilityScope | None = None
    instruction_state: RepositoryInstructionState = Field(default_factory=RepositoryInstructionState)
    file_observations: dict[str, FileObservation] = Field(default_factory=dict)
    completion_requirements: CompletionRequirements | None = None
    permission_grants: list[PermissionGrant] = Field(default_factory=list)
    last_status: Literal["running", "completed", "blocked", "failed", "cancelled"] | None = None
    last_reason: str | None = None


class SessionSnapshot(ContractModel):
    metadata: SessionMetadata
    messages: list[RuntimeMessage] = Field(default_factory=list)
    runtime_state: SessionRuntimeState = Field(default_factory=SessionRuntimeState)


class SessionOverview(FrozenContractModel):
    session_id: str = Field(min_length=1)
    model: str | None = None
    repository_root: str | None = None
    updated_at: str = Field(min_length=1)
    status: Literal[
        "unknown",
        "running",
        "completed",
        "blocked",
        "failed",
        "cancelled",
        "invalid",
    ]
    working_directory: str | None = None
    worktree: WorktreeSession | None = None
    error: str | None = None
