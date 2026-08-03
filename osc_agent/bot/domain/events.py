"""定义 Bot 可靠投递事件。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from osc_agent.bot.domain.jobs import utc_now
from osc_agent.contracts import ContractModel


class OutboxEvent(ContractModel):
    event_id: str = Field(min_length=1)
    job_id: str = Field(min_length=36, max_length=36)
    kind: str = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=200)
    payload: dict[str, object]
    status: Literal["pending", "processing", "completed", "failed", "dead_letter"] = "pending"
    attempts: int = Field(default=0, ge=0)
    next_attempt_at: str = Field(default_factory=utc_now)
    last_error: str | None = None
