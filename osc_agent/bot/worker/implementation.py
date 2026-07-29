"""执行已审批的 Bot Implementation Conversation。"""

from __future__ import annotations

from uuid import uuid4

from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.execution import validate_implementation_approval
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotCatalog
from osc_agent.bot.persistence.session_store import SqliteSessionStore
from osc_agent.bot.persistence.store import BotStore
from osc_agent.bot.worker.application import build_implementation_application
from osc_agent.bot.worker.context import resolve_job_context
from osc_agent.bot.worker.conversation import consume_runtime_events
from osc_agent.bot.worker.docker_runner import DockerProcessRunner
from osc_agent.bot.worker.inputs import build_implementation_skill_input
from osc_agent.configuration import AgentSettings
from osc_agent.runtime.gateway import ModelGateway


class ImplementationJobExecutor:
    def __init__(
        self,
        *,
        settings: AgentSettings,
        bot_settings: BotWorkerSettings,
        catalog: RepositoryBotCatalog,
        store: BotStore,
        sessions: SqliteSessionStore,
        model_gateway: ModelGateway | None,
    ) -> None:
        self.settings = settings
        self.bot_settings = bot_settings
        self.catalog = catalog
        self.store = store
        self.sessions = sessions
        self.model_gateway = model_gateway

    async def execute(self, job: BotJob) -> None:
        resolved = resolve_job_context(
            job,
            settings=self.settings,
            bot_settings=self.bot_settings,
            catalog=self.catalog,
            store=self.store,
        )
        config, contract, workspace = resolved.repository, resolved.contract, resolved.workspace
        approval = self.store.get_approval(job.approval_id or "")
        plan = self.store.get_plan_artifact(job.plan_artifact_id or "")
        validate_implementation_approval(approval, plan)
        assert plan is not None
        session_id = job.implementation_session_id or str(uuid4())
        if job.implementation_session_id is not None:
            persisted = self.sessions.load(job.implementation_session_id)
            if (
                persisted is not None
                and persisted.runtime_state.last_status == "completed"
                and self.store.get_delivery_draft(job.job_id) is not None
            ):
                self._finish_implementation(job.job_id)
                return
        runner = DockerProcessRunner(
            workspace_root=self.bot_settings.workspace_root,
            image_id=job.image_id,
            repository_config=config,
            job_id=job.job_id,
            cancel_check=lambda: self._job_cancelled(job.job_id),
            policy_violation_callback=lambda reason: self._terminate_repository_policy(
                job.job_id, reason
            ),
        )
        app = build_implementation_application(
            settings=self.settings,
            bot_settings=self.bot_settings,
            store=self.store,
            sessions=self.sessions,
            model_gateway=self.model_gateway,
            job=job,
            repository_config=config,
            contract=contract,
            workspace=workspace,
            process_runner=runner,
            on_policy_violation=lambda reason: self._terminate_repository_policy(
                job.job_id, reason
            ),
        )
        conversation = app.open_session(session_id)
        if job.implementation_session_id is None:
            current = self.store.get_job(job.job_id)
            assert current is not None
            self.store.update_job_fields(current.job_id, expected_version=current.version, implementation_session_id=session_id)
        initial_input = None
        if conversation.snapshot() is None:
            initial_input = build_implementation_skill_input(
                job,
                contract,
                plan,
                skill_name=self.catalog.implementation_skill_name,
            )
        query = conversation.submit(initial_input)
        completed = await consume_runtime_events(
            query,
            store=self.store,
            worker_id=self.bot_settings.worker_id,
            job_id=job.job_id,
            phase="implementation",
        )
        if not completed or self.store.get_delivery_draft(job.job_id) is None:
            raise ValueError("implementation Session did not satisfy the delivery contract")
        self._finish_implementation(job.job_id)

    def _finish_implementation(self, job_id: str) -> None:
        current = self.store.get_job(job_id)
        assert current is not None
        if current.status not in {"running_implementation", "queued_implementation"}:
            return
        self.store.transition_with_outbox(
            job_id=current.job_id,
            expected_version=current.version,
            status="ready_to_publish",
            lease_owner=None,
            lease_until=None,
            event=OutboxEvent(
                event_id=str(uuid4()),
                job_id=current.job_id,
                kind="publish",
                idempotency_key=f"publish:{current.job_id}",
                payload={},
            ),
        )

    def _job_cancelled(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        return job is None or job.status == "cancelled"

    def _terminate_repository_policy(self, job_id: str, reason: str) -> None:
        current = self.store.get_job(job_id)
        if current is None or current.status in {"completed", "stale", "dead_letter", "cancelled"}:
            return
        self.store.transition(
            job_id=job_id,
            expected_version=current.version,
            status="cancelled",
            error_code="REPOSITORY_POLICY_VIOLATION",
            error_message=reason[:1_000],
            lease_owner=None,
            lease_until=None,
        )

