"""验证Bot 安全边界的契约、边界条件与回归行为。"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import subprocess
import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from osc_agent.bot.config import BotControlSettings
from osc_agent.bot.control.github import GitHubAppClient
from osc_agent.bot.control.publisher import TrustedPublisher
from osc_agent.bot.control.webhook import create_webhook_app, verify_webhook_signature
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.execution import ExecutionContract
from osc_agent.bot.domain.repositories import RepositoryBotConfig
from osc_agent.bot.persistence.store import BotStore
from osc_agent.bot.worker.docker_runner import DockerProcessRunner
from osc_agent.processes.contracts import ProcessRequest
from osc_agent.completion.models import CompletionRequirements


IMAGE_ID = "sha256:" + "a" * 64


def test_github_installation_tokens_are_scoped_by_purpose(tmp_path: Path) -> None:
    key = tmp_path / "key.pem"
    key.write_text("unused", encoding="utf-8")
    repositories = tmp_path / "repositories.yml"
    repositories.write_text("repositories: {}\n", encoding="utf-8")
    settings = BotControlSettings(
        github_app_id=1,
        github_app_private_key_path=key,
        github_webhook_secret="x" * 16,
        github_public_url="https://agent.example.com",
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=repositories,
        github_commit_name="Bot",
        github_commit_email="bot@example.com",
        model_id="model",
    )
    client = GitHubAppClient(settings)
    requested: list[dict[str, object]] = []

    async def issue_token(_method, _path, *, json_body=None):
        requested.append(json_body)
        return {"token": f"token-{len(requested)}", "expires_at": "2099-01-01T00:00:00Z"}

    client._app_request = issue_token  # type: ignore[method-assign]
    purposes = (
        "repository_read",
        "issue_read",
        "issue_write",
        "pull_request_read",
        "pull_request_write",
        "contents_write",
    )
    for purpose in purposes:
        asyncio.run(client.installation_token(1, purpose=purpose))  # type: ignore[arg-type]

    assert requested == [
        {"permissions": {"contents": "read"}},
        {"permissions": {"issues": "read"}},
        {"permissions": {"issues": "write"}},
        {"permissions": {"pull_requests": "read"}},
        {"permissions": {"pull_requests": "write"}},
        {"permissions": {"contents": "write"}},
    ]


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
    import osc_agent.bot.worker.docker_runner as sandbox_module

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
    import osc_agent.bot.worker.docker_runner as sandbox_module

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


def test_docker_runner_logs_the_process_invocation_id(tmp_path: Path, monkeypatch) -> None:
    import osc_agent.bot.worker.docker_runner as runner_module

    class Process:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

        def kill(self) -> None:
            self.returncode = -1

    async def create_process(*_args, **_kwargs):
        return Process()

    logged: dict[str, object] = {}
    monkeypatch.setattr(runner_module.sys, "platform", "linux")
    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner_module, "git_workspace_fingerprint", lambda **_kwargs: "fingerprint")
    monkeypatch.setattr(runner_module, "git_snapshot", lambda **_kwargs: {"files": [], "patch": ""})
    monkeypatch.setattr(
        runner_module,
        "log_event",
        lambda _logger, _event, **fields: logged.update(fields),
    )
    repository = tmp_path / "jobs" / "job" / "implementation"
    (repository / ".git").mkdir(parents=True)
    runner = DockerProcessRunner(
        workspace_root=tmp_path / "jobs",
        image_id=IMAGE_ID,
        repository_config=RepositoryBotConfig(
            image=IMAGE_ID,
            validation_commands=("python -m pytest",),
        ),
        job_id="job-id",
        docker_executable="docker",
    )

    result = asyncio.run(
        runner.run(
            ProcessRequest(
                invocation_id="tool-use-1",
                executable="bash",
                command="python -m pytest",
                repo_root=str(repository),
                timeout_seconds=30,
                environment={},
            )
        )
    )

    assert result.exit_code == 0
    assert logged["tool_use_id"] == "tool-use-1"


def test_publisher_reuses_persisted_completion_requirements(
    monkeypatch, tmp_path: Path
) -> None:
    import osc_agent.bot.control.publisher as publisher_module

    requirements = CompletionRequirements(
        required_evidence=frozenset({"issue_plan"})
    )
    snapshot = SimpleNamespace(
        messages=[],
        state=SimpleNamespace(
            last_status="completed",
            completion_requirements=requirements,
        ),
    )
    captured = None

    class Evaluator:
        async def evaluate(self, evaluation):
            nonlocal captured
            captured = evaluation
            return SimpleNamespace(blocking_reasons=())

    monkeypatch.setattr(
        publisher_module,
        "SqliteSessionStore",
        lambda _store: SimpleNamespace(load=lambda _session_id: snapshot),
    )
    monkeypatch.setattr(publisher_module, "CompletionEvaluator", Evaluator)
    contract = ExecutionContract(
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=1,
        base_branch="main",
        base_sha="a" * 40,
        issue_number=1,
        issue_input_hash="b" * 64,
        model_id="model",
        plan_allowed_tools=frozenset({"read_file"}),
        implementation_allowed_tools=frozenset({"read_file"}),
        validation_commands=("python -m pytest",),
        denied_paths=(".git/**",),
        max_changed_files=10,
        max_patch_bytes=10_000,
        image_id=IMAGE_ID,
        command_timeout_seconds=60,
        container_cpus=1,
        container_memory="1g",
        container_pids=64,
    )
    job = BotJob(
        job_id=str(uuid4()),
        repository_id=contract.repository_id,
        repository_full_name=contract.repository_full_name,
        installation_id=contract.installation_id,
        issue_number=contract.issue_number,
        issue_url="https://github.com/owner/repo/issues/1",
        base_branch=contract.base_branch,
        base_sha=contract.base_sha,
        issue_input_hash=contract.issue_input_hash,
        execution_contract_hash=contract.contract_hash,
        image_id=contract.image_id,
        status="publishing",
        implementation_session_id="session",
    )
    publisher = object.__new__(TrustedPublisher)
    publisher.store = object()  # type: ignore[assignment]

    asyncio.run(publisher._verify_completion(job, contract, tmp_path))

    assert captured.requirements == requirements
    assert captured.validation_commands == contract.validation_commands


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
    settings = BotControlSettings(
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
        model_id="model",
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
        implementation_workspace_path=str(repo),
        implementation_workspace_ready=True,
    )
    asyncio.run(publisher._verify_persisted_commit(job, "fix: issue", repo))
    (repo / "untracked.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the persisted commit"):
        asyncio.run(publisher._verify_persisted_commit(job, "fix: issue", repo))
