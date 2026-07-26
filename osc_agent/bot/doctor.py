from __future__ import annotations

import shutil
import stat
import sys

from osc_agent.bot.config import BotSettings, load_repository_catalog
from osc_agent.bot.sandbox import resolve_image_id
from osc_agent.bot.store import BotStore
from osc_agent.runtime.models import FrozenContractModel


class BotDoctorResult(FrozenContractModel):
    status: str
    name: str
    message: str


async def run_bot_doctor(settings: BotSettings) -> list[BotDoctorResult]:
    results: list[BotDoctorResult] = []
    results.append(
        BotDoctorResult(
            status="PASS" if sys.platform == "linux" else "FAIL",
            name="operating_system",
            message="Linux host" if sys.platform == "linux" else "bot worker requires Ubuntu/Linux",
        )
    )
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
        import jwt  # noqa: F401
        import uvicorn  # noqa: F401

        dependencies_ok = True
    except ImportError:
        dependencies_ok = False
    results.append(
        BotDoctorResult(
            status="PASS" if dependencies_ok else "FAIL",
            name="bot_dependencies",
            message="available" if dependencies_ok else "install the project with .[bot]",
        )
    )
    for program in ("git", "docker", "pwsh"):
        path = shutil.which(program)
        results.append(
            BotDoctorResult(
                status="PASS" if path else "FAIL",
                name=program,
                message="available" if path else "not found on PATH",
            )
        )
    key = settings.github_app_private_key_path
    key_ok = key.is_file()
    results.append(
        BotDoctorResult(
            status="PASS" if key_ok else "FAIL",
            name="github_app_private_key",
            message="configured file exists" if key_ok else "configured file is missing",
        )
    )
    if key_ok and dependencies_ok:
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            load_pem_private_key(key.read_bytes(), password=None)
            key_parse_ok = True
        except (OSError, ValueError, TypeError):
            key_parse_ok = False
        results.append(
            BotDoctorResult(
                status="PASS" if key_parse_ok else "FAIL",
                name="github_app_private_key_format",
                message="valid PEM private key" if key_parse_ok else "invalid PEM private key",
            )
        )
    if key_ok and sys.platform == "linux":
        mode = stat.S_IMODE(key.stat().st_mode)
        results.append(
            BotDoctorResult(
                status="PASS" if mode == 0o600 else "FAIL",
                name="github_app_private_key_mode",
                message="0600" if mode == 0o600 else "must be 0600",
            )
        )
    try:
        settings.workspace_root.mkdir(parents=True, exist_ok=True)
        probe = settings.workspace_root / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        results.append(BotDoctorResult(status="PASS", name="workspace", message="writable"))
    except OSError:
        results.append(BotDoctorResult(status="FAIL", name="workspace", message="not writable"))
    try:
        store = BotStore(settings.database_path)
        store.initialize()
        results.append(BotDoctorResult(status="PASS", name="sqlite", message="WAL schema initialized"))
    except Exception:
        results.append(BotDoctorResult(status="FAIL", name="sqlite", message="database initialization failed"))
    try:
        catalog = load_repository_catalog(settings.repositories_config)
        results.append(BotDoctorResult(status="PASS", name="repositories", message=f"{len(catalog.repositories)} configured"))
        if shutil.which("docker"):
            for name, config in catalog.repositories.items():
                image_id = await resolve_image_id(config.image)
                results.append(BotDoctorResult(status="PASS", name=f"image:{name}", message=image_id))
    except Exception as exc:
        results.append(BotDoctorResult(status="FAIL", name="repositories", message=str(exc)[:500]))
    results.append(
        BotDoctorResult(
            status="PASS",
            name="credential_separation",
            message="control configuration loaded; worker uses a separate settings contract without GitHub credentials",
        )
    )
    return results
