"""定义无重复运行字段的 Session V5 持久化模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.state import AgentRunState
from osc_agent.workspaces.models import WorktreeSession


class SessionMetadata(FrozenContractModel):
    schema_version: Literal[5]
    session_id: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt: str = ""


class SessionSnapshot(ContractModel):
    metadata: SessionMetadata
    messages: list[RuntimeMessage] = Field(default_factory=list)
    state: AgentRunState


class SessionOverview(FrozenContractModel):
    session_id: str = Field(min_length=1)
    model: str | None = None
    workspace_root: str | None = None
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
