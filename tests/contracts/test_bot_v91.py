from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from osc_agent.bot.config import BotSettings, BotWorkerSettings
from osc_agent.bot.doctor import run_bot_control_doctor, run_bot_worker_doctor
from osc_agent.bot.models import BotJob, RepositoryBotCatalog, RepositoryBotConfig
from osc_agent.bot.worker import BotWorker
from osc_agent.cli import app
from osc_agent.config import Settings


IMAGE_ID = "sha256:" + "d" * 64


def _catalog_file(tmp_path: Path) -> Path:
    path = tmp_path / "repositories.yml"
    path.write_text(
        "repositories:\n"
        "  owner/repo:\n"
        f"    image: {IMAGE_ID}\n"
        "    validation_commands: [python -m pytest]\n",
        encoding="utf-8",
    )
    return path


def _worker_settings(tmp_path: Path) -> BotWorkerSettings:
    return BotWorkerSettings(
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=_catalog_file(tmp_path),
        worker_id="worker-1",
    )


def test_control_doctor_never_checks_docker(monkeypatch, tmp_path: Path) -> None:
    import osc_agent.bot.doctor as doctor_module

    checked: list[str] = []
    monkeypatch.setattr(doctor_module.sys, "platform", "linux")
    monkeypatch.setattr(doctor_module, "_bot_dependencies_available", lambda: False)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda program: checked.append(program) or f"/usr/bin/{program}",
    )
    settings = BotSettings(
        github_app_id=1,
        github_app_private_key_path=tmp_path / "missing.pem",
        github_webhook_secret="x" * 16,
        github_public_url="https://agent.example.com",
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=_catalog_file(tmp_path),
        github_commit_name="OSA Bot",
        github_commit_email="bot@example.com",
    )

    results = asyncio.run(run_bot_control_doctor(settings))

    assert checked == ["git"]
    assert not {"docker", "bash", "rg"} & set(checked)
    assert next(item for item in results if item.name == "docker_separation").status == "PASS"


def test_worker_doctor_checks_rg_and_exact_image(monkeypatch, tmp_path: Path) -> None:
    import osc_agent.bot.doctor as doctor_module

    checked: list[str] = []
    inspected: list[str] = []
    monkeypatch.setattr(doctor_module.sys, "platform", "linux")
    monkeypatch.setattr(doctor_module, "_bot_dependencies_available", lambda: True)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda program: checked.append(program) or f"/usr/bin/{program}",
    )

    async def resolve(image: str) -> str:
        inspected.append(image)
        return image

    monkeypatch.setattr(doctor_module, "resolve_image_id", resolve)
    results = asyncio.run(
        run_bot_worker_doctor(
            _worker_settings(tmp_path),
            Settings(anthropic_api_key="secret", model_id="model"),
        )
    )

    assert {"git", "docker", "bash", "rg"} <= set(checked)
    assert inspected == [IMAGE_ID]
    assert next(item for item in results if item.name == "image:owner/repo").status == "PASS"


def test_worker_doctor_fails_when_rg_or_image_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    import osc_agent.bot.doctor as doctor_module

    monkeypatch.setattr(doctor_module.sys, "platform", "linux")
    monkeypatch.setattr(doctor_module, "_bot_dependencies_available", lambda: True)
    monkeypatch.setattr(
        doctor_module.shutil,
        "which",
        lambda program: None if program == "rg" else f"/usr/bin/{program}",
    )

    async def mismatch(_image: str) -> str:
        return "sha256:" + "e" * 64

    monkeypatch.setattr(doctor_module, "resolve_image_id", mismatch)
    results = asyncio.run(
        run_bot_worker_doctor(
            _worker_settings(tmp_path),
            Settings(anthropic_api_key="secret", model_id="model"),
        )
    )

    assert next(item for item in results if item.name == "rg").status == "FAIL"
    assert any(
        item.name == "repositories" and item.status == "FAIL" for item in results
    )


def test_plan_worker_never_resolves_docker_image(monkeypatch, tmp_path: Path) -> None:
    import osc_agent.bot.worker as worker_module

    job = BotJob(
        job_id=str(uuid4()),
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=1,
        issue_number=1,
        issue_url="https://github.com/owner/repo/issues/1",
        base_sha="a" * 40,
        image_id=IMAGE_ID,
        status="running_plan",
        plan_workspace_path=str(tmp_path / "workspaces" / "job"),
        plan_workspace_ready=True,
    )

    class Store:
        transitioned: dict[str, object] | None = None

        def claim_job(self, _worker_id):
            return job

        def get_job(self, _job_id):
            return job

        def transition_with_outbox(self, **changes):
            self.transitioned = changes

    async def must_not_resolve(_image: str) -> str:
        raise AssertionError("Plan must not inspect Docker")

    async def complete_plan(*_args, **_kwargs):
        return None

    monkeypatch.setattr(worker_module, "resolve_image_id", must_not_resolve)
    monkeypatch.setattr(BotWorker, "_run_plan", complete_plan)
    store = Store()
    worker = BotWorker(
        settings=Settings(anthropic_api_key="secret", model_id="model"),
        bot_settings=_worker_settings(tmp_path),
        catalog=RepositoryBotCatalog(
            repositories={
                "owner/repo": RepositoryBotConfig(
                    image=IMAGE_ID,
                    validation_commands=("python -m pytest",),
                )
            }
        ),
        store=store,  # type: ignore[arg-type]
    )

    assert asyncio.run(worker.run_once()) is True
    assert store.transitioned is None


def test_bot_doctor_requires_exactly_one_role() -> None:
    runner = CliRunner()

    assert runner.invoke(app, ["bot", "doctor"]).exit_code != 0
    assert runner.invoke(app, ["bot", "doctor", "--control", "--worker"]).exit_code != 0


def test_python_worker_image_template_is_offline_ready() -> None:
    dockerfile = (
        Path(__file__).parents[2] / "deploy" / "docker" / "python311" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "python3-venv" in dockerfile
    assert "ripgrep" in dockerfile
    assert "ENTRYPOINT []" in dockerfile
    assert 'CMD ["/bin/bash"' in dockerfile
