"""定义 Control 解析后的严格 GitHub 命令上下文。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, HttpUrl

from osc_agent.contracts import FrozenContractModel


class WebhookCommand(FrozenContractModel):
    action: Literal["plan", "implement", "reply", "status", "retry", "cancel"]
    job_id: str | None = None
    message: str | None = None


class IssueCommentContext(FrozenContractModel):
    installation_id: int = Field(gt=0)
    repository_id: int = Field(gt=0)
    repository: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    issue_number: int = Field(gt=0)
    issue_url: HttpUrl
    body: str = Field(min_length=1)
    actor_id: int = Field(gt=0)
    actor_login: str = Field(min_length=1)
    comment_id: int = Field(gt=0)
