"""领取并执行规划或实现阶段的 Bot 作业。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4

from osc_agent.bot.application_factory import (
    build_implementation_application,
    build_plan_application,
)
from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.agent_inputs import (
    build_implementation_skill_input,
    build_planning_skill_input,
)
from osc_agent.bot.models import (
    OutboxEvent,
    RepositoryBotCatalog,
    validate_implementation_approval,
)
from osc_agent.bot.sandbox import DockerProcessRunner, resolve_image_id
from osc_agent.bot.store import BotStore, SqliteSessionStore
from osc_agent.configuration import AgentSettings
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.runtime.events import (
    RunCompleted,
    RunStopped,
)


class BotModelContractMismatch(ValueError):
    """执行合同绑定的模型与 Worker 实际模型不一致。"""


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
        self.settings = settings
        self.bot_settings = bot_settings
        self.catalog = catalog
        self.store = store
        self.model_gateway = model_gateway
        self.sessions = SqliteSessionStore(store)

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
                await self._run_plan(job)
            else:
                resolved_image_id = await resolve_image_id(job.image_id)
                if resolved_image_id != job.image_id:
                    raise ValueError("configured immutable Docker image does not match the local image")
                await self._run_implementation(job)
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

    async def _run_plan(self, job) -> None:
        config, contract, workspace = self._workspace(job)
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
        completed = await self._consume(query, job.job_id, phase="plan")
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

    async def _run_implementation(self, job) -> None:
        config, contract, workspace = self._workspace(job)
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
        completed = await self._consume(query, job.job_id, phase="implementation")
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

    async def _consume(
        self,
        events,
        job_id: str,
        *,
        phase: Literal["plan", "implementation"],
    ) -> bool:
        completed = False
        last_heartbeat = monotonic()
        last_progress = 0.0
        iterator = events.__aiter__()
        pending = asyncio.create_task(iterator.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({pending}, timeout=5)
                current = self.store.get_job(job_id)
                if current is None or current.status == "cancelled":
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    raise ValueError("bot job was cancelled while the Agent was running")
                if monotonic() - last_heartbeat >= 20:
                    self.store.heartbeat(job_id, self.bot_settings.worker_id)
                    last_heartbeat = monotonic()
                if not done:
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    break
                event_type = event.type
                now = monotonic()
                if (
                    last_progress == 0.0
                    or now - last_progress >= 5
                    or isinstance(event, (RunCompleted, RunStopped))
                ):
                    self.store.record_progress(job_id, f"{phase}:{event_type}")
                    last_progress = now
                if isinstance(event, RunCompleted):
                    completed = True
                elif isinstance(event, RunStopped):
                    completed = False
                pending = asyncio.create_task(iterator.__anext__())
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        return completed

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

    def _workspace(self, job):
        configured = self.catalog.repositories.get(job.repository_full_name)
        contract = self.store.get_execution_contract(job.execution_contract_hash)
        plan_phase = job.status == "running_plan"
        workspace_value = job.plan_workspace_path if plan_phase else job.implementation_workspace_path
        ready = job.plan_workspace_ready if plan_phase else job.implementation_workspace_ready
        if configured is None or not configured.enabled or contract is None or not workspace_value or not ready:
            raise ValueError("job has no enabled repository configuration or prepared workspace")
        if (
            contract.contract_hash != job.execution_contract_hash
            or contract.repository_id != job.repository_id
            or contract.repository_full_name != job.repository_full_name
            or contract.base_sha != job.base_sha
            or contract.issue_input_hash != job.issue_input_hash
        ):
            raise ValueError("Job does not match its execution contract")
        if not self.settings.model_id or contract.model_id != self.settings.model_id:
            raise BotModelContractMismatch(
                "execution contract model does not match the configured Worker model"
            )
        config = type(configured)(
            enabled=True,
            image=contract.image_id,
            validation_commands=contract.validation_commands,
            denied_paths=contract.denied_paths,
            max_changed_files=contract.max_changed_files,
            max_patch_bytes=contract.max_patch_bytes,
            command_timeout_seconds=contract.command_timeout_seconds,
            container_cpus=contract.container_cpus,
            container_memory=contract.container_memory,
            container_pids=contract.container_pids,
            pull_request_mode=contract.pull_request_mode,
        )
        workspace = Path(workspace_value).resolve()
        if not workspace.is_relative_to(self.bot_settings.workspace_root.resolve()):
            raise ValueError("prepared workspace escapes the bot workspace root")
        return config, contract, workspace

    def _comment_event(self, job_id: str, suffix: str, body: str) -> OutboxEvent:
        return OutboxEvent(
            event_id=str(uuid4()),
            job_id=job_id,
            kind="issue_comment",
            idempotency_key=f"comment:{job_id}:{suffix}",
            payload={"body": body + f"\n\n<!-- osa-job:{job_id}:{suffix} -->"},
        )
