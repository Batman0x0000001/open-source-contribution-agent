"""领取 Bot Job、分派阶段并处理重试与 Dead Letter。"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.repositories import RepositoryBotCatalog
from osc_agent.bot.persistence.session_store import SqliteSessionStore
from osc_agent.bot.persistence.store import BotStore
from osc_agent.bot.worker.context import BotModelContractMismatch
from osc_agent.bot.worker.docker_runner import resolve_image_id
from osc_agent.bot.worker.implementation import ImplementationJobExecutor
from osc_agent.bot.worker.plan import PlanJobExecutor
from osc_agent.configuration import AgentSettings
from osc_agent.runtime.gateway import ModelGateway


class BotWorker:
    def __init__(
        self,
        *,
        settings: AgentSettings,
        bot_settings: BotWorkerSettings,
        catalog: RepositoryBotCatalog,
        store: BotStore,
        model_gateway: ModelGateway | None = None,
    ) -> None:
        self.bot_settings = bot_settings
        self.store = store
        self.sessions = SqliteSessionStore(store)
        self.plan_executor = PlanJobExecutor(
            settings=settings,
            bot_settings=bot_settings,
            catalog=catalog,
            store=store,
            sessions=self.sessions,
            model_gateway=model_gateway,
        )
        self.implementation_executor = ImplementationJobExecutor(
            settings=settings,
            bot_settings=bot_settings,
            catalog=catalog,
            store=store,
            sessions=self.sessions,
            model_gateway=model_gateway,
        )

    async def run_once(
        self,
        phase: Literal["plan", "implementation"] | None = None,
    ) -> bool:
        job = (
            self.store.claim_job(self.bot_settings.worker_id, phase=phase)
            if phase is not None
            else self.store.claim_job(self.bot_settings.worker_id)
        )
        if job is None:
            return False
        try:
            if job.status == "running_plan":
                await self.plan_executor.execute(job)
            else:
                resolved_image_id = await resolve_image_id(job.image_id)
                if resolved_image_id != job.image_id:
                    raise ValueError("configured immutable Docker image does not match the local image")
                await self.implementation_executor.execute(job)
        except BotModelContractMismatch as exc:
            current = self.store.get_job(job.job_id)
            if current is not None and current.status in {"running_plan", "running_implementation"}:
                retry_phase = "plan" if current.status == "running_plan" else "implementation"
                retrying = self.store.transition_with_outbox(
                    job_id=current.job_id,
                    expected_version=current.version,
                    status="retry_wait",
                    retry_phase=retry_phase,
                    event=self._comment_event(
                        current.job_id,
                        "model-contract-mismatch",
                        "The Bot job cannot run because its approved model contract no longer matches the Worker configuration.",
                    ),
                    error_code="BOT_MODEL_CONTRACT_MISMATCH",
                    error_message=str(exc)[:1_000],
                    lease_owner=None,
                    lease_until=None,
                )
                self.store.transition(
                    job_id=retrying.job_id,
                    expected_version=retrying.version,
                    status="dead_letter",
                )
        except Exception as exc:
            current = self.store.get_job(job.job_id)
            if current is not None and current.status not in {"cancelled", "completed", "stale"}:
                retry_phase = "plan" if current.status == "running_plan" else "implementation"
                retrying = self.store.transition_with_outbox(
                    job_id=current.job_id,
                    expected_version=current.version,
                    status="retry_wait",
                    retry_phase=retry_phase,
                    event=self._comment_event(
                        current.job_id,
                        "failed",
                        "The bot job failed inside the trusted worker. Review server audit logs before retrying.",
                    ),
                    error_code="BOT_WORKER_FAILED",
                    error_message=str(exc)[:1_000],
                    lease_owner=None,
                    lease_until=None,
                )
                attempts = (
                    retrying.plan_attempts
                    if retry_phase == "plan"
                    else retrying.implementation_attempts
                )
                if attempts >= 3:
                    self.store.transition(
                        job_id=retrying.job_id,
                        expected_version=retrying.version,
                        status="dead_letter",
                    )
        return True

    def _comment_event(self, job_id: str, suffix: str, body: str) -> OutboxEvent:
        return OutboxEvent(
            event_id=str(uuid4()),
            job_id=job_id,
            kind="issue_comment",
            idempotency_key=f"comment:{job_id}:{suffix}",
            payload={"body": body + f"\n\n<!-- osa-job:{job_id}:{suffix} -->"},
        )
