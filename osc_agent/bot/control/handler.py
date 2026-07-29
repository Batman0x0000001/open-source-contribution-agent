"""处理 GitHub 命令、作业编排和控制面状态转换。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from uuid import UUID, uuid4

from osc_agent.bot.config import BotControlSettings
from osc_agent.bot.control.commands import parse_issue_comment, parse_webhook_command
from osc_agent.bot.control.github import GitHubControlClient
from osc_agent.bot.control.models import IssueCommentContext
from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.execution import BotApproval, ExecutionContract, plan_evidence_hash
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotCatalog, RepositoryBotConfig
from osc_agent.bot.persistence.store import BotStore


AUTHORIZED_PERMISSIONS = {"write", "maintain", "admin"}
class BotControlService:
    def __init__(
        self,
        *,
        settings: BotControlSettings,
        catalog: RepositoryBotCatalog,
        store: BotStore,
        github: GitHubControlClient,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.store = store
        self.github = github

    async def handle_issue_comment(self, payload: dict[str, object]) -> str:
        if payload.get("action") != "created":
            return "ignored"
        issue = payload.get("issue")
        sender = payload.get("sender")
        if isinstance(issue, dict) and issue.get("pull_request") is not None:
            return "ignored"
        if isinstance(sender, dict):
            login = sender.get("login")
            if sender.get("type") == "Bot" or (isinstance(login, str) and login.endswith("[bot]")):
                return "ignored"
        parsed = parse_issue_comment(payload)
        command = parse_webhook_command(parsed.body)
        if command is None:
            return "ignored"
        repository = parsed.repository
        config = self.catalog.repositories.get(repository)
        if config is None or not config.enabled:
            raise ValueError("repository is not enabled for this GitHub App bot")
        permission = await self.github.collaborator_permission(
            parsed.installation_id, repository, parsed.actor_login
        )
        if permission not in AUTHORIZED_PERMISSIONS:
            raise PermissionError("comment author does not have repository write permission")
        if command.action == "plan":
            return await self._plan(parsed, config)
        if command.job_id is not None:
            UUID(command.job_id)
        job = (
            self.store.get_job(command.job_id)
            if command.job_id is not None
            else self.store.get_active_job(int(parsed.repository_id), int(parsed.issue_number))
        )
        if job is None or job.repository_full_name != repository or job.issue_number != parsed.issue_number:
            raise ValueError("job does not belong to this repository issue")
        if command.action == "implement":
            return await self._implement(job, parsed)
        if command.action == "reply":
            assert command.message is not None
            source_id = f"github:{parsed.comment_id}"
            self.store.enqueue_plan_reply(
                source_id=source_id,
                job=job,
                body=command.message,
                prepare_event=OutboxEvent(
                    event_id=str(uuid4()), job_id=job.job_id, kind="prepare",
                    idempotency_key=f"prepare:{job.job_id}:plan:{parsed.comment_id}",
                    payload={"phase": "plan"},
                ),
                acknowledgement_event=self._comment_event(
                    job, f"reply-received:{parsed.comment_id}",
                    "Plan reply received. The original planning Session will resume.",
                ),
            )
            return job.job_id
        if command.action == "status":
            self.store.enqueue_outbox(self._comment_event(job, "status", f"Job `{job.job_id}` is `{job.status}`."))
            return job.status
        if command.action == "retry":
            if job.status != "retry_wait":
                raise ValueError("job is not waiting for retry")
            target = {"plan": "queued_plan", "implementation": "queued_implementation", "publish": "ready_to_publish"}.get(job.retry_phase or "")
            if target is None:
                raise ValueError("job retry phase is missing")
            self.store.transition(
                job_id=job.job_id, expected_version=job.version, status=target, retry_phase=None
            )
            return job.job_id
        return self._cancel(job)

    async def _plan(self, parsed: IssueCommentContext, config: RepositoryBotConfig) -> str:
        existing = self.store.get_active_job(int(parsed.repository_id), int(parsed.issue_number))
        if existing is not None:
            return existing.job_id
        branch, base_sha = await self.github.repository_head(
            int(parsed.installation_id), str(parsed.repository)
        )
        issue_evidence = await self.github.issue(
            int(parsed.installation_id), str(parsed.repository), int(parsed.issue_number)
        )
        issue_hash = sha256(json.dumps(issue_evidence, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        contract = ExecutionContract(
            repository_id=int(parsed.repository_id), repository_full_name=str(parsed.repository),
            installation_id=int(parsed.installation_id), base_branch=branch, base_sha=base_sha,
            issue_number=int(parsed.issue_number), issue_input_hash=issue_hash, model_id=self.settings.model_id,
            plan_allowed_tools=frozenset({"read_file","glob","grep","git_status","git_diff","git_log","read_tool_result","agent","submit_issue_plan"}),
            implementation_allowed_tools=frozenset({"read_file","glob","grep","bash","git_status","git_diff","git_log","read_skill_resource","write_file","edit_file","read_tool_result","agent","submit_delivery_draft"}),
            validation_commands=config.validation_commands, denied_paths=config.denied_paths,
            max_changed_files=config.max_changed_files, max_patch_bytes=config.max_patch_bytes,
            image_id=config.image, command_timeout_seconds=config.command_timeout_seconds,
            container_cpus=config.container_cpus, container_memory=config.container_memory,
            container_pids=config.container_pids, pull_request_mode=config.pull_request_mode,
        )
        self.store.save_execution_contract(contract)
        job = BotJob(
            job_id=str(uuid4()),
            repository_id=int(parsed.repository_id),
            repository_full_name=str(parsed.repository),
            installation_id=int(parsed.installation_id),
            issue_number=int(parsed.issue_number),
            issue_url=str(parsed.issue_url),
            base_branch=branch, base_sha=base_sha, issue_input_hash=issue_hash,
            execution_contract_hash=contract.contract_hash, image_id=config.image,
            status="queued_plan",
        )
        self.store.create_job(
            job,
            input_value=issue_evidence,
            outbox_events=(
                OutboxEvent(
                    event_id=str(uuid4()), job_id=job.job_id, kind="prepare",
                    idempotency_key=f"prepare:{job.job_id}:plan", payload={"phase": "plan"},
                ),
                self._comment_event(
                    job, "plan-queued",
                    f"Plan job `{job.job_id}` was queued. Use `/osa status` for progress.",
                ),
            ),
        )
        return job.job_id

    async def _implement(self, job: BotJob, parsed: IssueCommentContext) -> str:
        if job.status != "waiting_approval" or job.plan_artifact_id is None:
            raise ValueError("job is not waiting for implementation approval")
        _branch, current_sha = await self.github.repository_head(job.installation_id, job.repository_full_name)
        if current_sha != job.base_sha:
            self.store.transition_with_outbox(
                job_id=job.job_id,
                expected_version=job.version,
                status="stale",
                event=OutboxEvent(
                    event_id=str(uuid4()),
                    job_id=job.job_id,
                    kind="issue_comment",
                    idempotency_key=f"comment:{job.job_id}:stale",
                    payload={
                        "body": "The default branch changed after planning. Run `/osa plan` again."
                        f"\n\n<!-- osa-job:{job.job_id}:stale -->"
                    },
                ),
            )
            return "stale"
        plan = self.store.get_plan_artifact(job.plan_artifact_id)
        if plan is None or plan.status != "ready":
            raise ValueError("job has no approved ready plan")
        if plan.execution_contract_hash != job.execution_contract_hash or plan.base_sha != job.base_sha:
            raise ValueError("plan artifact does not match the Job execution contract")
        evidence_hash = plan_evidence_hash(plan)
        approval = BotApproval(
            approval_id=str(uuid4()),
            job_id=job.job_id,
            actor_id=int(parsed.actor_id),
            actor_login=str(parsed.actor_login),
            evidence_hash=evidence_hash,
            execution_contract_hash=job.execution_contract_hash,
            base_sha=job.base_sha,
            expires_at=(
                datetime.now(timezone.utc) + timedelta(days=self.settings.plan_approval_days)
            ).isoformat(),
        )
        updated = self.store.approve_implementation(
            job_id=job.job_id,
            expected_version=job.version,
            approval=approval,
            outbox_event=OutboxEvent(
                event_id=str(uuid4()),
                job_id=job.job_id,
                kind="prepare",
                idempotency_key=f"prepare:{job.job_id}:implementation",
                payload={"phase": "implementation"},
            ),
        )
        return updated.job_id

    def _cancel(self, job: BotJob) -> str:
        if job.status in {"completed", "cancelled"}:
            return job.status
        self.store.transition_with_outbox(
            job_id=job.job_id,
            expected_version=job.version,
            status="cancelled",
            lease_owner=None,
            lease_until=None,
            event=OutboxEvent(
                event_id=str(uuid4()),
                job_id=job.job_id,
                kind="issue_comment",
                idempotency_key=f"comment:{job.job_id}:cancelled",
                payload={
                    "body": f"Job `{job.job_id}` was cancelled.\n\n<!-- osa-job:{job.job_id}:cancelled -->"
                },
            ),
        )
        return "cancelled"

    def _comment_event(self, job: BotJob, suffix: str, body: str) -> OutboxEvent:
        return OutboxEvent(event_id=str(uuid4()), job_id=job.job_id, kind="issue_comment",
            idempotency_key=f"comment:{job.job_id}:{suffix}:{job.version}", payload={"body": body})


