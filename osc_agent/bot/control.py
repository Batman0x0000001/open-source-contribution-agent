from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from uuid import UUID, uuid4

from osc_agent.bot.config import BotSettings
from osc_agent.bot.github_app import GitHubControlClient
from osc_agent.bot.models import (
    BotApproval,
    BotJob,
    OutboxEvent,
    RepositoryBotCatalog,
    WebhookCommand,
    plan_evidence_hash,
)
from osc_agent.bot.store import BotStore


AUTHORIZED_PERMISSIONS = {"write", "maintain", "admin"}
_JOB_ID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_IMPLEMENT = re.compile(rf"^/osa implement ({_JOB_ID})$")
_CANCEL = re.compile(rf"^/osa cancel ({_JOB_ID})$")


def parse_webhook_command(body: str) -> WebhookCommand | None:
    command = body.strip()
    if command == "/osa plan":
        return WebhookCommand(action="plan")
    if match := _IMPLEMENT.fullmatch(command):
        return WebhookCommand(action="implement", job_id=match.group(1))
    if match := _CANCEL.fullmatch(command):
        return WebhookCommand(action="cancel", job_id=match.group(1))
    return None


class BotControlService:
    def __init__(
        self,
        *,
        settings: BotSettings,
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
        parsed = _parse_issue_comment(payload)
        command = parse_webhook_command(parsed["body"])
        if command is None:
            return "ignored"
        repository = parsed["repository"]
        config = self.catalog.repositories.get(repository)
        if config is None or not config.enabled:
            raise ValueError("repository is not enabled for this GitHub App bot")
        permission = await self.github.collaborator_permission(
            parsed["installation_id"], repository, parsed["actor_login"]
        )
        if permission not in AUTHORIZED_PERMISSIONS:
            raise PermissionError("comment author does not have repository write permission")
        if command.action == "plan":
            return await self._plan(parsed, config.image)
        if command.job_id is None:
            raise ValueError("job id is required")
        UUID(command.job_id)
        job = self.store.get_job(command.job_id)
        if job is None or job.repository_full_name != repository or job.issue_number != parsed["issue_number"]:
            raise ValueError("job does not belong to this repository issue")
        if command.action == "implement":
            return await self._implement(job, parsed)
        return self._cancel(job)

    async def _plan(self, parsed: dict[str, object], image: str) -> str:
        _branch, base_sha = await self.github.repository_head(
            int(parsed["installation_id"]), str(parsed["repository"])
        )
        job = BotJob(
            job_id=str(uuid4()),
            repository_id=int(parsed["repository_id"]),
            repository_full_name=str(parsed["repository"]),
            installation_id=int(parsed["installation_id"]),
            issue_number=int(parsed["issue_number"]),
            issue_url=str(parsed["issue_url"]),
            base_sha=base_sha,
            image_id=image,
            status="queued_plan",
        )
        issue_evidence = await self.github.issue(
            job.installation_id, job.repository_full_name, job.issue_number
        )
        self.store.create_job(
            job,
            input_value=issue_evidence,
            outbox_event=OutboxEvent(
                event_id=str(uuid4()),
                job_id=job.job_id,
                kind="prepare",
                idempotency_key=f"prepare:{job.job_id}:plan",
                payload={"phase": "plan"},
            ),
        )
        return job.job_id

    async def _implement(self, job: BotJob, parsed: dict[str, object]) -> str:
        if job.status != "waiting_implementation" or job.plan_artifact_id is None:
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
        evidence_hash = plan_evidence_hash(plan)
        approval = BotApproval(
            approval_id=str(uuid4()),
            job_id=job.job_id,
            actor_id=int(parsed["actor_id"]),
            actor_login=str(parsed["actor_login"]),
            evidence_hash=evidence_hash,
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


def _parse_issue_comment(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("action") != "created":
        raise ValueError("only newly created issue comments are supported")
    installation = payload.get("installation")
    repository = payload.get("repository")
    issue = payload.get("issue")
    comment = payload.get("comment")
    sender = payload.get("sender")
    if not all(isinstance(item, dict) for item in (installation, repository, issue, comment, sender)):
        raise ValueError("GitHub issue_comment payload is incomplete")
    assert isinstance(installation, dict) and isinstance(repository, dict)
    assert isinstance(issue, dict) and isinstance(comment, dict) and isinstance(sender, dict)
    if issue.get("pull_request") is not None:
        raise ValueError("pull request comments are not supported")
    login = sender.get("login")
    if sender.get("type") == "Bot" or (isinstance(login, str) and login.endswith("[bot]")):
        raise ValueError("bot comments are ignored")
    values = {
        "installation_id": installation.get("id"),
        "repository_id": repository.get("id"),
        "repository": repository.get("full_name"),
        "issue_number": issue.get("number"),
        "issue_url": issue.get("html_url"),
        "body": comment.get("body"),
        "actor_id": sender.get("id"),
        "actor_login": login,
    }
    if not isinstance(values["installation_id"], int) or not isinstance(values["repository_id"], int):
        raise ValueError("GitHub installation or repository id is invalid")
    if not isinstance(values["issue_number"], int) or not isinstance(values["actor_id"], int):
        raise ValueError("GitHub issue or actor id is invalid")
    if any(not isinstance(values[key], str) or not values[key] for key in ("repository", "issue_url", "body", "actor_login")):
        raise ValueError("GitHub issue_comment text fields are invalid")
    return values
