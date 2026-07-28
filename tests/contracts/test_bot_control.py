from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from osc_agent.bot.config import BotSettings
from osc_agent.bot.control import BotControlService, parse_webhook_command
from osc_agent.bot.models import IssuePlanArtifact, RepositoryBotCatalog, RepositoryBotConfig
from osc_agent.bot.store import BotStore


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


def _settings(tmp_path: Path) -> BotSettings:
    key = tmp_path / "app.pem"
    key.write_text("private", encoding="utf-8")
    repositories = tmp_path / "repositories.yml"
    repositories.write_text("repositories: {}\n", encoding="utf-8")
    return BotSettings(
        github_app_id=1,
        github_app_private_key_path=key,
        github_webhook_secret="x" * 16,
        github_public_url="https://agent.example.com",
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=repositories,
        worker_id="worker-1",
        github_commit_name="OSA Bot",
        github_commit_email="bot@example.com",
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
    assert parse_webhook_command(" /osa plan\n").action == "plan"
    job_id = str(uuid4())
    assert parse_webhook_command(f"/osa implement {job_id}").job_id == job_id
    assert parse_webhook_command("/OSA plan") is None
    assert parse_webhook_command("/osa plan now") is None
    assert parse_webhook_command("text /osa plan") is None


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
    job_id = asyncio.run(control.handle_issue_comment(_payload("/osa plan")))
    job = store.get_job(job_id)
    assert job is not None and job.status == "queued_plan"
    assert job.image_id == IMAGE_ID
    assert store.get_job_input(job_id)["trust"] == "untrusted_external"
    plan = IssuePlanArtifact(
        status="ready",
        base_sha=job.base_sha,
        execution_contract_hash=job.execution_contract_hash,
        summary="summary",
        plan_markdown="approved plan",
    )
    artifact_id = store.save_artifact(job_id, plan)
    running = store.transition(
        job_id=job_id, expected_version=job.version, status="running_plan"
    )
    waiting = store.transition(
        job_id=job_id,
        expected_version=running.version,
        status="waiting_approval",
        plan_artifact_id=artifact_id,
    )
    result = asyncio.run(control.handle_issue_comment(_payload(f"/osa implement {job_id}")))
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
        asyncio.run(control.handle_issue_comment(_payload("/osa plan")))
