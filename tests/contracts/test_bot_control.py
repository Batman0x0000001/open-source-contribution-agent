"""验证Bot 控制面的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from osc_agent.bot.config import BotControlSettings
from osc_agent.bot.control.commands import parse_webhook_command
from osc_agent.bot.control.handler import BotControlService
from osc_agent.bot.domain.artifacts import IssuePlanArtifact
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotCatalog, RepositoryBotConfig
from osc_agent.bot.domain.state_machine import JobEvent
from osc_agent.bot.persistence.store import BotStore


IMAGE_ID = "sha256:" + "b" * 64


class FakeGitHub:
    permission = "write"
    head = "a" * 40

    async def collaborator_permission(self, installation_id, repository, login):
        return self.permission

    async def repository_head(self, installation_id, repository):
        return "main", self.head

    async def issue(self, installation_id, repository, number):
        return {"content_source": "github", "trust": "untrusted_external", "issue": {"title": "Bug"}, "comments": []}


def _settings(tmp_path: Path) -> BotControlSettings:
    key = tmp_path / "app.pem"
    key.write_text("private", encoding="utf-8")
    repositories = tmp_path / "repositories.yml"
    repositories.write_text("repositories: {}\n", encoding="utf-8")
    return BotControlSettings(
        github_app_id=1,
        github_app_private_key_path=key,
        github_webhook_secret="x" * 16,
        github_public_url="https://agent.example.com",
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=repositories,
        worker_id="worker-1",
        github_commit_name="osc-agent",
        github_commit_email="bot@example.com",
        model_id="model",
    )


def _payload(body: str) -> dict[str, object]:
    return {
        "action": "created",
        "installation": {"id": 10},
        "repository": {"id": 20, "full_name": "owner/repo"},
        "issue": {"number": 7, "html_url": "https://github.com/owner/repo/issues/7"},
        "comment": {"id": 40, "body": body},
        "sender": {"id": 30, "login": "maintainer", "type": "User"},
    }


def test_webhook_command_uses_exact_input() -> None:
    assert parse_webhook_command(" /osc-agent plan\n").action == "plan"
    job_id = str(uuid4())
    assert parse_webhook_command(f"/osc-agent implement {job_id}").job_id == job_id
    assert parse_webhook_command(f"/osc-agent cancel {job_id}").job_id == job_id
    assert parse_webhook_command("/osc-agent status").action == "status"
    assert parse_webhook_command("/osc-agent retry").action == "retry"
    assert parse_webhook_command("/osc-agent reply more context").message == "more context"
    assert parse_webhook_command("/OSC-AGENT plan") is None
    assert parse_webhook_command("/osc-agent plan now") is None
    assert parse_webhook_command("text /osc-agent plan") is None
    assert parse_webhook_command("/osc plan") is None


def test_control_creates_plan_and_bound_implementation_approval(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = BotStore(settings.database_path)
    store.initialize()
    github = FakeGitHub()
    control = BotControlService(
        settings=settings,
        catalog=RepositoryBotCatalog(
            repositories={
                "owner/repo": RepositoryBotConfig(
                    image=IMAGE_ID, validation_commands=("python -m pytest",)
                )
            }
        ),
        store=store,
        github=github,
    )
    job_id = asyncio.run(control.handle_issue_comment(_payload("/osc-agent plan")))
    job = store.get_job(job_id)
    assert job is not None and job.status == "queued_plan"
    assert job.image_id == IMAGE_ID
    contract = store.get_execution_contract(job.execution_contract_hash)
    assert contract is not None and contract.model_id == settings.model_id
    assert store.get_job_input(job_id)["trust"] == "untrusted_external"
    for comment_id in (41, 42):
        status_payload = _payload("/osc-agent status")
        status_payload["comment"] = {"id": comment_id, "body": "/osc-agent status"}
        assert asyncio.run(control.handle_issue_comment(status_payload)) == "queued_plan"
    with store.connect() as connection:
        status_keys = connection.execute(
            "SELECT idempotency_key FROM outbox_events "
            "WHERE idempotency_key LIKE ? ORDER BY idempotency_key",
            (f"comment:{job_id}:status:%",),
        ).fetchall()
    assert len(status_keys) == 2
    plan = IssuePlanArtifact(
        status="ready",
        base_sha=job.base_sha,
        execution_contract_hash=job.execution_contract_hash,
        summary="summary",
        plan_markdown="approved plan",
    )
    artifact_id = store.save_artifact(job_id, plan)
    running = store.apply_job_event(
        job_id=job_id, expected_version=job.version, event=JobEvent.CLAIM_PLAN
    )
    store.apply_job_event(
        job_id=job_id,
        expected_version=running.version,
        event=JobEvent.PLAN_READY,
        plan_artifact_id=artifact_id,
    )
    result = asyncio.run(control.handle_issue_comment(_payload(f"/osc-agent implement {job_id}")))
    approved = store.get_job(result)
    assert approved is not None and approved.status == "queued_implementation"
    assert approved.approval_id is not None
    assert store.get_approval(approved.approval_id).actor_login == "maintainer"


def test_control_rejects_non_writer(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = BotStore(settings.database_path)
    store.initialize()
    github = FakeGitHub()
    github.permission = "read"
    control = BotControlService(
        settings=settings,
        catalog=RepositoryBotCatalog(
            repositories={
                "owner/repo": RepositoryBotConfig(
                    image=IMAGE_ID, validation_commands=("python -m pytest",)
                )
            }
        ),
        store=store,
        github=github,
    )
    with pytest.raises(PermissionError):
        asyncio.run(control.handle_issue_comment(_payload("/osc-agent plan")))


@pytest.mark.parametrize("status", ["completed", "stale", "dead_letter", "cancelled"])
def test_cancel_is_idempotent_for_every_terminal_status(status: str) -> None:
    control = object.__new__(BotControlService)
    job = BotJob(
        job_id=str(uuid4()),
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=1,
        issue_number=1,
        issue_url="https://github.com/owner/repo/issues/1",
        base_sha="a" * 40,
        image_id=IMAGE_ID,
        status=status,
    )

    assert control._cancel(job) == status


def test_cancel_rejects_the_non_atomic_publishing_window() -> None:
    control = object.__new__(BotControlService)
    job = BotJob(
        job_id=str(uuid4()),
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=1,
        issue_number=1,
        issue_url="https://github.com/owner/repo/issues/1",
        base_sha="a" * 40,
        image_id=IMAGE_ID,
        status="publishing",
    )

    with pytest.raises(ValueError, match="can no longer be cancelled"):
        control._cancel(job)
