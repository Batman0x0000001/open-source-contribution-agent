"""验证Bot 端到端流程的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from osc_agent.bot.config import BotControlSettings, BotMaintenanceSettings, BotWorkerSettings
from osc_agent.bot.control.doctor import run_bot_control_doctor
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotCatalog, RepositoryBotConfig
from osc_agent.bot.worker.coordinator import BotWorker
from osc_agent.bot.worker.doctor import run_bot_worker_doctor
from osc_agent.bot.entrypoint import app as bot_app
from tests.settings_factory import make_agent_settings as Settings


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
    import osc_agent.bot.diagnostics as diagnostics_module

    checked: list[str] = []
    monkeypatch.setattr(diagnostics_module.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics_module, "bot_dependencies_available", lambda: False)
    monkeypatch.setattr(
        diagnostics_module.shutil,
        "which",
        lambda program: checked.append(program) or f"/usr/bin/{program}",
    )
    settings = BotControlSettings(
        github_app_id=1,
        github_app_private_key_path=tmp_path / "missing.pem",
        github_webhook_secret="x" * 16,
        github_public_url="https://agent.example.com",
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=_catalog_file(tmp_path),
        github_commit_name="OSA Bot",
        github_commit_email="bot@example.com",
        model_id="model",
    )

    results = asyncio.run(run_bot_control_doctor(settings))

    assert checked == ["git"]
    assert not {"docker", "bash", "rg"} & set(checked)
    assert next(item for item in results if item.name == "docker_separation").status == "PASS"


def test_worker_doctor_checks_rg_and_exact_image(monkeypatch, tmp_path: Path) -> None:
    import osc_agent.bot.diagnostics as diagnostics_module
    import osc_agent.bot.worker.doctor as doctor_module

    checked: list[str] = []
    inspected: list[str] = []
    monkeypatch.setattr(diagnostics_module.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics_module, "bot_dependencies_available", lambda: True)
    monkeypatch.setattr(
        diagnostics_module.shutil,
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
    import osc_agent.bot.diagnostics as diagnostics_module
    import osc_agent.bot.worker.doctor as doctor_module

    monkeypatch.setattr(diagnostics_module.sys, "platform", "linux")
    monkeypatch.setattr(diagnostics_module, "bot_dependencies_available", lambda: True)
    monkeypatch.setattr(
        diagnostics_module.shutil,
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
    import osc_agent.bot.worker.coordinator as worker_module

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
    worker.plan_executor = SimpleNamespace(execute=complete_plan)  # type: ignore[assignment]

    assert asyncio.run(worker.run_once()) is True
    assert store.transitioned is None


def test_bot_doctor_requires_exactly_one_role() -> None:
    runner = CliRunner()

    assert runner.invoke(bot_app, ["doctor"]).exit_code != 0
    assert runner.invoke(bot_app, ["doctor", "--control", "--worker"]).exit_code != 0


def test_bot_entrypoint_exposes_only_bot_service_and_operations() -> None:
    result = CliRunner().invoke(bot_app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "control",
        "worker",
        "doctor",
        "cleanup",
        "schema-check",
        "archive-state",
        "reset-state",
        "smoke-test",
        "render-state-machine",
    ):
        assert command in result.output
    assert "contribute" not in result.output


def test_bot_maintenance_settings_do_not_require_github_credentials(tmp_path: Path) -> None:
    settings = BotMaintenanceSettings(
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
    )

    assert settings.database_path == tmp_path / "bot.sqlite3"


def test_bot_maintenance_settings_reject_database_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces"

    with pytest.raises(ValueError, match="outside"):
        BotMaintenanceSettings(
            database_path=workspace / "bot.sqlite3",
            workspace_root=workspace,
        )


def test_python_worker_image_template_is_offline_ready() -> None:
    dockerfile = (
        Path(__file__).parents[2] / "deploy" / "docker" / "python311" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "python3-venv" in dockerfile
    assert "ripgrep" in dockerfile
    assert "ENTRYPOINT []" in dockerfile
    assert 'CMD ["/bin/bash"' in dockerfile


def test_ubuntu_deployment_uses_accessible_maintenance_flag_and_valid_unit_sections() -> None:
    root = Path(__file__).parents[2]
    nginx = (root / "deploy" / "ubuntu" / "nginx.conf.example").read_text(encoding="utf-8")
    upgrade = (root / "deploy" / "ubuntu" / "upgrade.sh").read_text(encoding="utf-8")
    control_unit = (
        root / "deploy" / "ubuntu" / "osc-agent-bot-control.service"
    ).read_text(encoding="utf-8")
    unit_section, service_section = control_unit.split("[Service]", maxsplit=1)

    assert "/run/osc-agent-webhook-maintenance" in nginx
    assert "/run/osc-agent-webhook-maintenance" in upgrade
    assert "/etc/osc-agent/webhook-maintenance" not in nginx + upgrade
    assert "StartLimitIntervalSec=120" in unit_section
    assert "StartLimitBurst=5" in unit_section
    assert "StartLimitIntervalSec" not in service_section
    assert "StartLimitBurst" not in service_section
    assert "osc-agent-bot schema-check" in control_unit
    assert "osc-agent-bot control" in control_unit
    assert "osc-agent deploy" not in upgrade + control_unit
    assert "osc-agent bot" not in upgrade + control_unit


def test_reset_state_sets_shared_runtime_permissions(monkeypatch, tmp_path: Path) -> None:
    database = tmp_path / "bot.sqlite3"
    workspace = tmp_path / "workspaces"
    chmod_calls: list[tuple[Path, int]] = []
    real_chmod = Path.chmod

    def record_chmod(target: Path, mode: int, *, follow_symlinks: bool = True) -> None:
        chmod_calls.append((target, mode))
        real_chmod(target, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "chmod", record_chmod)
    monkeypatch.setenv("OSC_AGENT_BOT_DATABASE_PATH", str(database))
    monkeypatch.setenv("OSC_AGENT_BOT_WORKSPACE_ROOT", str(workspace))

    result = CliRunner().invoke(bot_app, ["reset-state", "--confirm"])

    assert result.exit_code == 0, result.output
    assert (database.resolve(), 0o660) in chmod_calls
    assert (workspace.resolve(), 0o2770) in chmod_calls
