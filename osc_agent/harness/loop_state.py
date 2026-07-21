from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from osc_agent.config import Settings
from osc_agent.harness.contracts import RunMetrics
from osc_agent.harness.progress import ProgressGuard
from osc_agent.harness.recovery import ModelRequestRecoveryState, ResponseRecoveryState


@dataclass
class LoopExecutionState:
    """单次 agent loop 的短生命周期可变状态。"""

    request_recovery: ModelRequestRecoveryState
    response_recovery: ResponseRecoveryState
    progress: ProgressGuard
    metrics: RunMetrics = field(default_factory=RunMetrics)
    started_at: float = field(default_factory=monotonic)
    budget_overrides: set[str] = field(default_factory=set)
    stopped: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "LoopExecutionState":
        return cls(
            request_recovery=ModelRequestRecoveryState(
                current_model=settings.model_id,
                fallback_model_id=settings.fallback_model_id,
            ),
            response_recovery=ResponseRecoveryState(),
            progress=ProgressGuard(
                repeat_action_limit=settings.repeat_action_limit,
                consecutive_failure_limit=settings.consecutive_failure_limit,
                no_progress_limit=settings.no_progress_limit,
            ),
        )

    def update_elapsed(self) -> None:
        self.metrics.elapsed_ms = int((monotonic() - self.started_at) * 1000)
