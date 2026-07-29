"""执行 Bot Plan Conversation 并持久化严格规划产物。"""

from __future__ import annotations

from uuid import uuid4

from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotCatalog
from osc_agent.bot.persistence.session_store import SqliteSessionStore
from osc_agent.bot.persistence.store import BotStore
from osc_agent.bot.worker.application import build_plan_application
from osc_agent.bot.worker.context import resolve_job_context
from osc_agent.bot.worker.conversation import consume_runtime_events
from osc_agent.bot.worker.inputs import build_planning_skill_input
from osc_agent.configuration import AgentSettings
from osc_agent.runtime.gateway import ModelGateway


class PlanJobExecutor:
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
        existing_artifact = self.store.latest_plan_artifact(job.job_id)
        if existing_artifact is not None:
            await self._finish_plan(job.job_id, *existing_artifact)
            return
        session_id = job.plan_session_id or str(uuid4())
        app = build_plan_application(
            settings=self.settings,
            bot_settings=self.bot_settings,
            store=self.store,
            sessions=self.sessions,
            model_gateway=self.model_gateway,
            job=job,
            repository_config=config,
            contract=contract,
            workspace=workspace,
        )
        conversation = app.open_session(session_id)
        if job.plan_session_id is None:
            current = self.store.get_job(job.job_id)
            assert current is not None
            self.store.update_job_fields(current.job_id, expected_version=current.version, plan_session_id=session_id)
        evidence = self.store.get_job_input(job.job_id)
        pending_reply = self.store.pending_plan_reply(job.job_id)
        if pending_reply is not None:
            self.store.append_external_message_once(
                source_id=pending_reply[0], session_id=session_id, text=pending_reply[1]
            )
        initial_input = None
        if conversation.snapshot() is None:
            initial_input = build_planning_skill_input(
                job,
                contract,
                evidence,
                skill_name=self.catalog.planning_skill_name,
            )
        query = conversation.submit(initial_input)
        completed = await consume_runtime_events(
            query,
            store=self.store,
            worker_id=self.bot_settings.worker_id,
            job_id=job.job_id,
            phase="plan",
        )
        artifact = self.store.latest_plan_artifact(job.job_id)
        if not completed or artifact is None:
            raise ValueError("planning Session ended without a strict plan artifact")
        artifact_id, plan = artifact
        await self._finish_plan(job.job_id, artifact_id, plan)

    async def _finish_plan(self, job_id, artifact_id, plan) -> None:
        current = self.store.get_job(job_id)
        assert current is not None
        if current.status not in {"running_plan", "queued_plan"}:
            return
        status = "waiting_approval" if plan.status == "ready" else "blocked_plan"
        body = (
            f"{plan.plan_markdown}\n\nJob ID: `{current.job_id}`\nBase: `{current.base_sha}`"
            if plan.status == "ready"
            else "Planning is blocked:\n\n" + "\n".join(f"- {item}" for item in plan.unresolved_questions)
        )
        self.store.transition_with_outbox(
            job_id=current.job_id,
            expected_version=current.version,
            status=status,
            event=self._comment_event(current.job_id, "plan", body),
            plan_artifact_id=artifact_id,
            lease_owner=None,
            lease_until=None,
        )

    def _comment_event(self, job_id: str, suffix: str, body: str) -> OutboxEvent:
        return OutboxEvent(
            event_id=str(uuid4()),
            job_id=job_id,
            kind="issue_comment",
            idempotency_key=f"comment:{job_id}:{suffix}",
            payload={"body": body + f"\n\n<!-- osa-job:{job_id}:{suffix} -->"},
        )

