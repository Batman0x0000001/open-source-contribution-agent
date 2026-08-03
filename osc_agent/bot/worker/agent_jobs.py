"""顺序实现 Bot Plan 与 Implementation 两条可信 Agent Job 用例。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4

from osc_agent.application import (
    AgentApplication,
    AgentApplicationConfig,
    AgentConversation,
    AgentProfile,
    ProductSkillInput,
    build_agent_application,
)
from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.domain.artifacts import IssuePlanArtifact
from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.execution import (
    ExecutionContract,
    repository_config_from_contract,
    validate_implementation_approval,
    validate_job_contract,
)
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotCatalog, RepositoryBotConfig
from osc_agent.bot.domain.state_machine import BotJobStateMachine, JobEvent
from osc_agent.bot.persistence.session_store import SqliteSessionStore
from osc_agent.bot.persistence.store import BotStore
from osc_agent.bot.worker.artifact_tools import SubmitDeliveryDraftTool, SubmitIssuePlanTool
from osc_agent.bot.worker.docker_runner import DockerProcessRunner
from osc_agent.bot.worker.policy import BotPermissionPolicy, BotRepositoryPolicyHook
from osc_agent.configuration import AgentSettings
from osc_agent.processes.contracts import ProcessRunner
from osc_agent.processes.runner import DisabledProcessRunner
from osc_agent.runtime.events import RunCompleted, RunStopped, RuntimeEvent
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.subagents.builtins import build_explore_subagent


class BotModelContractMismatch(ValueError):
    """执行合同绑定的模型与 Worker 实际模型不一致。"""


class BotJobLeaseLost(ValueError):
    """当前 Worker 已不再拥有 Job 或 Runtime Session Lease。"""


@dataclass(frozen=True)
class _ResolvedJobContext:
    repository: RepositoryBotConfig
    contract: ExecutionContract
    workspace: Path


def _resolve_job_context(
    job: BotJob,
    *,
    settings: AgentSettings,
    bot_settings: BotWorkerSettings,
    catalog: RepositoryBotCatalog,
    store: BotStore,
) -> _ResolvedJobContext:
    configured = catalog.repositories.get(job.repository_full_name)
    contract = store.get_execution_contract(job.execution_contract_hash)
    planning = job.status == "running_plan"
    workspace_value = job.plan_workspace_path if planning else job.implementation_workspace_path
    ready = job.plan_workspace_ready if planning else job.implementation_workspace_ready
    if configured is None or not configured.enabled or contract is None or not workspace_value or not ready:
        raise ValueError("job has no enabled repository configuration or prepared workspace")
    validate_job_contract(job, contract)
    if not settings.model_id or contract.model_id != settings.model_id:
        raise BotModelContractMismatch("execution contract model does not match the configured Worker model")
    repository = repository_config_from_contract(contract)
    workspace = Path(workspace_value).resolve()
    if not workspace.is_relative_to(bot_settings.workspace_root.resolve()):
        raise ValueError("prepared workspace escapes the bot workspace root")
    return _ResolvedJobContext(repository, contract, workspace)


def build_planning_skill_input(
    job: BotJob,
    contract: ExecutionContract,
    issue_evidence: Mapping[str, object],
) -> ProductSkillInput:
    validate_job_contract(job, contract)
    return ProductSkillInput(
        name=contract.planning_skill_name,
        arguments={
            "issue_evidence": dict(issue_evidence),
            "base_sha": job.base_sha,
            "execution_contract_hash": job.execution_contract_hash,
        },
    )


def build_implementation_skill_input(
    job: BotJob,
    contract: ExecutionContract,
    plan: IssuePlanArtifact,
) -> ProductSkillInput:
    validate_job_contract(job, contract)
    if plan.base_sha != job.base_sha or plan.execution_contract_hash != job.execution_contract_hash:
        raise ValueError("approved plan does not match the Bot Job contract")
    return ProductSkillInput(
        name=contract.implementation_skill_name,
        arguments={
            "repo_url": f"https://github.com/{job.repository_full_name}",
            "goal": f"Implement approved plan for issue #{job.issue_number}",
            "mode": "approved_implementation",
            "automation": {
                "issue_number": job.issue_number,
                "base_sha": job.base_sha,
                "approved_plan": plan.plan_markdown,
                "execution_contract_hash": job.execution_contract_hash,
            },
        },
    )


async def consume_runtime_events(
    events,
    *,
    store: BotStore,
    sessions: SqliteSessionStore,
    session_id: str,
    worker_id: str,
    job_id: str,
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
            current = store.get_job(job_id)
            if (
                current is None
                or current.status == "cancelled"
                or current.lease_owner != worker_id
            ):
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                raise BotJobLeaseLost(
                    "bot job was cancelled or reclaimed while the Agent was running"
                )
            if monotonic() - last_heartbeat >= 20:
                store.heartbeat(job_id, worker_id)
                sessions.renew_lease(session_id)
                last_heartbeat = monotonic()
            if not done:
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            now = monotonic()
            if (
                last_progress == 0.0
                or now - last_progress >= 5
                or isinstance(event, (RunCompleted, RunStopped))
            ):
                store.record_progress(job_id, worker_id, f"{phase}:{event.type}")
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


def _bot_session_events(
    conversation: AgentConversation,
    initial_input: ProductSkillInput,
) -> AsyncIterator[RuntimeEvent]:
    """保留 Job 已分配 Session ID、但 Session 尚未落库时的崩溃恢复语义。"""

    if conversation.snapshot() is None:
        return conversation.start(initial_input)
    return conversation.resume()


def _build_plan_application(
    *,
    settings: AgentSettings,
    bot_settings: BotWorkerSettings,
    store: BotStore,
    sessions: SqliteSessionStore,
    model_gateway: ModelGateway | None,
    job: BotJob,
    repository: RepositoryBotConfig,
    contract: ExecutionContract,
    workspace: Path,
) -> AgentApplication:
    return build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=workspace,
            profile=AgentProfile(
                profile_id="bot_plan",
                system_prompt=(
                    "You are the read-only planning component of an authenticated "
                    "GitHub App job."
                ),
                allowed_tools=contract.plan_allowed_tools,
                allowed_initial_skills=frozenset({contract.planning_skill_name}),
                required_evidence=frozenset({"issue_plan"}),
                start_in_plan_mode=True,
            ),
            model_gateway=model_gateway,
            session_store=sessions,
            state_root=bot_settings.workspace_root.parent / "runtime-state",
            process_runner=DisabledProcessRunner(),
            permission_policy=BotPermissionPolicy(implementation_approved=False),
            extra_tools=(
                SubmitIssuePlanTool(
                    store,
                    job_id=job.job_id,
                    worker_id=bot_settings.worker_id,
                    base_sha=job.base_sha,
                    execution_contract_hash=job.execution_contract_hash,
                ),
            ),
            pre_tool_hooks=(BotRepositoryPolicyHook(repository),),
            subagent_registrations=(
                build_explore_subagent(
                    model=settings.model_id or "",
                    config=settings.runtime.agents.explore.to_query_config(),
                ),
            ),
        )
    )


def _build_implementation_application(
    *,
    settings: AgentSettings,
    bot_settings: BotWorkerSettings,
    store: BotStore,
    sessions: SqliteSessionStore,
    model_gateway: ModelGateway | None,
    job: BotJob,
    repository: RepositoryBotConfig,
    contract: ExecutionContract,
    workspace: Path,
    process_runner: ProcessRunner,
    on_policy_violation: Callable[[str], None],
) -> AgentApplication:
    prompt = (
        "This is an approved GitHub App implementation job. Repository writes and "
        "process execution are allowed only through the provided tools; process commands "
        "run in a network-disabled Docker sandbox. Never ask questions, enter a worktree, "
        "commit, push, or access GitHub. Run every configured validation command exactly "
        "as written after the final modification:\n- "
        + "\n- ".join(repository.validation_commands)
    )
    return build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=workspace,
            profile=AgentProfile(
                profile_id="bot_implementation",
                system_prompt=prompt,
                allowed_tools=contract.implementation_allowed_tools,
                allowed_initial_skills=frozenset(
                    {contract.implementation_skill_name}
                ),
                required_evidence=frozenset(
                    {
                        "successful_test",
                        "independent_verification",
                        "git_change_snapshot",
                        "delivery_draft",
                    }
                ),
            ),
            model_gateway=model_gateway,
            session_store=sessions,
            state_root=bot_settings.workspace_root.parent / "runtime-state",
            process_runner=process_runner,
            permission_policy=BotPermissionPolicy(implementation_approved=True),
            validation_commands=repository.validation_commands,
            extra_tools=(
                SubmitDeliveryDraftTool(
                    store,
                    job_id=job.job_id,
                    worker_id=bot_settings.worker_id,
                    issue_number=job.issue_number,
                    base_sha=job.base_sha,
                    execution_contract_hash=job.execution_contract_hash,
                ),
            ),
            pre_tool_hooks=(
                BotRepositoryPolicyHook(
                    repository, on_violation=on_policy_violation
                ),
            ),
        )
    )


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
        self.settings, self.bot_settings, self.catalog = settings, bot_settings, catalog
        self.store, self.sessions, self.model_gateway = store, sessions, model_gateway

    async def execute(self, job: BotJob) -> None:
        resolved = _resolve_job_context(
            job,
            settings=self.settings,
            bot_settings=self.bot_settings,
            catalog=self.catalog,
            store=self.store,
        )
        existing = self.store.latest_plan_artifact(job.job_id)
        if existing is not None:
            self._finish_plan(job.job_id, *existing)
            return
        session_id = job.plan_session_id or str(uuid4())
        application = _build_plan_application(
            settings=self.settings,
            bot_settings=self.bot_settings,
            store=self.store,
            sessions=self.sessions,
            model_gateway=self.model_gateway,
            job=job,
            repository=resolved.repository,
            contract=resolved.contract,
            workspace=resolved.workspace,
        )
        conversation = application.open_session(session_id)
        if job.plan_session_id is None:
            current = self.store.get_job(job.job_id)
            if current is None:
                raise ValueError("Plan Job disappeared before its Session ID was persisted")
            self.store.update_job_fields(
                current.job_id,
                expected_version=current.version,
                required_lease_owner=self.bot_settings.worker_id,
                plan_session_id=session_id,
            )
        pending_reply = self.store.pending_plan_reply(job.job_id)
        if pending_reply is not None:
            self.store.append_external_message_once(
                source_id=pending_reply[0],
                session_id=session_id,
                text=pending_reply[1],
            )
        initial = build_planning_skill_input(
            job,
            resolved.contract,
            self.store.get_job_input(job.job_id),
        )
        completed = await consume_runtime_events(
            _bot_session_events(conversation, initial),
            store=self.store,
            sessions=self.sessions,
            session_id=session_id,
            worker_id=self.bot_settings.worker_id,
            job_id=job.job_id,
            phase="plan",
        )
        artifact = self.store.latest_plan_artifact(job.job_id)
        if not completed or artifact is None:
            raise ValueError("planning Session ended without a strict plan artifact")
        self._finish_plan(job.job_id, *artifact)

    def _finish_plan(
        self,
        job_id: str,
        artifact_id: str,
        plan: IssuePlanArtifact,
    ) -> None:
        current = self.store.get_job(job_id)
        if current is None:
            raise ValueError("Plan Job disappeared before completion")
        if current.status != "running_plan":
            return
        event = JobEvent.PLAN_READY if plan.status == "ready" else JobEvent.PLAN_BLOCKED
        body = (
            f"{plan.plan_markdown}\n\nJob ID: `{current.job_id}`\nBase: `{current.base_sha}`"
            if plan.status == "ready"
            else "Planning is blocked:\n\n"
            + "\n".join(f"- {item}" for item in plan.unresolved_questions)
        )
        self.store.apply_job_event_with_outbox(
            job_id=current.job_id,
            expected_version=current.version,
            event=event,
            required_lease_owner=self.bot_settings.worker_id,
            outbox_event=_comment_event(current.job_id, "plan", body),
            plan_artifact_id=artifact_id,
            lease_owner=None,
            lease_until=None,
        )


def _comment_event(job_id: str, suffix: str, body: str) -> OutboxEvent:
    return OutboxEvent(
        event_id=str(uuid4()),
        job_id=job_id,
        kind="issue_comment",
        idempotency_key=f"comment:{job_id}:{suffix}",
        payload={"body": body + f"\n\n<!-- osc-job:{job_id}:{suffix} -->"},
    )


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
        resolved = _resolve_job_context(
            job,
            settings=self.settings,
            bot_settings=self.bot_settings,
            catalog=self.catalog,
            store=self.store,
        )
        config, contract, workspace = resolved.repository, resolved.contract, resolved.workspace
        approval = self.store.get_approval(job.approval_id or "")
        plan = self.store.get_plan_artifact(job.plan_artifact_id or "")
        if plan is None:
            raise ValueError("Implementation Job has no approved Plan artifact")
        validate_implementation_approval(approval, plan)
        session_id = job.implementation_session_id or str(uuid4())
        if job.implementation_session_id is not None:
            persisted = self.sessions.load(job.implementation_session_id)
            if (
                persisted is not None
                and persisted.state.last_status == "completed"
                and self.store.get_delivery_draft(job.job_id) is not None
            ):
                self._finish_implementation(job.job_id)
                return
        runner = DockerProcessRunner(
            workspace_root=self.bot_settings.workspace_root,
            image_id=job.image_id,
            repository_config=config,
            job_id=job.job_id,
            cancel_check=lambda: self._job_should_stop(job.job_id),
            policy_violation_callback=lambda reason: self._terminate_repository_policy(
                job.job_id, reason
            ),
        )
        app = _build_implementation_application(
            settings=self.settings,
            bot_settings=self.bot_settings,
            store=self.store,
            sessions=self.sessions,
            model_gateway=self.model_gateway,
            job=job,
            repository=config,
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
            if current is None:
                raise ValueError("Implementation Job disappeared before its Session ID was persisted")
            self.store.update_job_fields(
                current.job_id,
                expected_version=current.version,
                required_lease_owner=self.bot_settings.worker_id,
                implementation_session_id=session_id,
            )
        initial_input = build_implementation_skill_input(
            job,
            contract,
            plan,
        )
        query = _bot_session_events(conversation, initial_input)
        completed = await consume_runtime_events(
            query,
            store=self.store,
            sessions=self.sessions,
            session_id=session_id,
            worker_id=self.bot_settings.worker_id,
            job_id=job.job_id,
            phase="implementation",
        )
        if not completed or self.store.get_delivery_draft(job.job_id) is None:
            raise ValueError("implementation Session did not satisfy the delivery contract")
        self._finish_implementation(job.job_id)

    def _finish_implementation(self, job_id: str) -> None:
        current = self.store.get_job(job_id)
        if current is None:
            raise ValueError("Implementation Job disappeared before completion")
        if current.status != "running_implementation":
            return
        self.store.apply_job_event_with_outbox(
            job_id=current.job_id,
            expected_version=current.version,
            event=JobEvent.IMPLEMENTATION_READY,
            required_lease_owner=self.bot_settings.worker_id,
            lease_owner=None,
            lease_until=None,
            outbox_event=OutboxEvent(
                event_id=str(uuid4()),
                job_id=current.job_id,
                kind="publish",
                idempotency_key=f"publish:{current.job_id}",
                payload={},
            ),
        )

    def _job_should_stop(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        return (
            job is None
            or job.status == "cancelled"
            or job.lease_owner != self.bot_settings.worker_id
        )

    def _terminate_repository_policy(self, job_id: str, reason: str) -> None:
        current = self.store.get_job(job_id)
        if current is None or BotJobStateMachine.is_terminal(current.status):
            return
        self.store.apply_job_event(
            job_id=job_id,
            expected_version=current.version,
            event=JobEvent.CANCEL,
            required_lease_owner=self.bot_settings.worker_id,
            error_code="REPOSITORY_POLICY_VIOLATION",
            error_message=reason[:1_000],
            lease_owner=None,
            lease_until=None,
        )
