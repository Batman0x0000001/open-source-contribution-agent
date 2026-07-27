from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import subprocess
import asyncio
from uuid import uuid4

import pytest

from osc_agent.bot.config import BotSettings
from osc_agent.bot.models import BotJob, RepositoryBotConfig
from osc_agent.bot.publisher import TrustedPublisher
from osc_agent.bot.sandbox import DockerProcessRunner
from osc_agent.bot.store import BotStore
from osc_agent.bot.webhook import create_webhook_app, verify_webhook_signature


IMAGE_ID = "sha256:" + "a" * 64


def test_webhook_signature_is_exact() -> None:
    body = b'{"action":"created"}'
    secret = "secret"
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(body, signature, secret)
    assert not verify_webhook_signature(body + b"x", signature, secret)
    assert not verify_webhook_signature(body, "sha1=bad", secret)


def test_webhook_endpoint_rejects_bad_signature_and_deduplicates(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    class Control:
        calls = 0

        async def handle_issue_comment(self, payload):
            self.calls += 1
            return "job-id"

    secret = "a sufficiently long secret"
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    control = Control()
    client = TestClient(create_webhook_app(secret=secret, store=store, control=control))
    body = json.dumps({"action": "created"}).encode()
    headers = {
        "X-Hub-Signature-256": "sha256="
        + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest(),
        "X-GitHub-Event": "issue_comment",
        "X-GitHub-Delivery": "delivery-1",
        "Content-Type": "application/json",
    }
    invalid = client.post(
        "/webhooks/github",
        content=body,
        headers={**headers, "X-Hub-Signature-256": "sha256=bad"},
    )
    assert invalid.status_code == 401
    first = client.post("/webhooks/github", content=body, headers=headers)
    second = client.post("/webhooks/github", content=body, headers=headers)
    assert first.status_code == 200 and first.json()["result"] == "job-id"
    assert second.status_code == 200 and second.json()["duplicate"] is True
    assert control.calls == 1


def test_docker_runner_builds_fail_closed_isolation_argv(tmp_path: Path, monkeypatch) -> None:
    import osc_agent.bot.sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module.sys, "platform", "linux")
    workspace = tmp_path / "jobs" / "job" / "implementation"
    (workspace / ".git").mkdir(parents=True)
    runner = DockerProcessRunner(
        workspace_root=tmp_path / "jobs",
        image_id="sha256:" + "a" * 64,
        repository_config=RepositoryBotConfig(
            image=IMAGE_ID, validation_commands=("python -m pytest",)
        ),
        job_id="job-id",
        docker_executable="docker",
    )
    argv = runner._argv("container", workspace, "python -m pytest")
    assert argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"][1].endswith("readonly")
    assert "ANTHROPIC_API_KEY" not in " ".join(argv)
    assert "GITHUB_TOKEN" not in " ".join(argv)


def test_docker_runner_rejects_mutable_image_and_non_linux(tmp_path: Path, monkeypatch) -> None:
    import osc_agent.bot.sandbox as sandbox_module

    config = RepositoryBotConfig(image=IMAGE_ID, validation_commands=("python -m pytest",))
    monkeypatch.setattr(sandbox_module.sys, "platform", "linux")
    with pytest.raises(ValueError, match="immutable"):
        DockerProcessRunner(
            workspace_root=tmp_path,
            image_id="worker:latest",
            repository_config=config,
            job_id="job-id",
            docker_executable="docker",
        )


def test_publisher_accepts_only_one_clean_commit_on_approved_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "file.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "--all"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
        cwd=repo,
        check=True,
    )
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "file.txt").write_text("changed", encoding="utf-8")
    subprocess.run(["git", "add", "--all"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fix: issue"],
        cwd=repo,
        check=True,
    )
    key = tmp_path / "key.pem"
    key.write_text("unused", encoding="utf-8")
    config = tmp_path / "repositories.yml"
    config.write_text("repositories: {}\n", encoding="utf-8")
    settings = BotSettings(
        github_app_id=1,
        github_app_private_key_path=key,
        github_webhook_secret="x" * 16,
        github_public_url="https://agent.example.com",
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=config,
        worker_id="worker-1",
        github_commit_name="Bot",
        github_commit_email="bot@example.com",
    )
    publisher = TrustedPublisher(settings=settings, store=BotStore(settings.database_path), github=object())
    job = BotJob(
        job_id=str(uuid4()),
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=1,
        issue_number=1,
        issue_url="https://github.com/owner/repo/issues/1",
        base_sha=base,
        image_id="sha256:" + "a" * 64,
        status="publishing",
        workspace_path=str(repo),
    )
    asyncio.run(publisher._verify_persisted_commit(job, "fix: issue", repo))
    (repo / "untracked.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the persisted commit"):
        asyncio.run(publisher._verify_persisted_commit(job, "fix: issue", repo))
