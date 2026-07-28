"""检查 Bot 控制面和 Worker 的部署依赖与配置。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import shutil
import stat
import sys

from osc_agent.bot.config import BotSettings, BotWorkerSettings, load_repository_catalog
from osc_agent.bot.sandbox import resolve_image_id
from osc_agent.bot.store import BotStore
from osc_agent.config import Settings
from osc_agent.runtime.models import FrozenContractModel


class BotDoctorResult(FrozenContractModel):
    status: str
    name: str
    message: str


async def run_bot_control_doctor(settings: BotSettings) -> list[BotDoctorResult]:
    """检查 Control 所需能力；该路径不得访问 Docker。"""

    results = _base_results(("git",))
    key = settings.github_app_private_key_path
    key_ok = key.is_file()
    results.append(
        BotDoctorResult(
            status="PASS" if key_ok else "FAIL",
            name="github_app_private_key",
            message="configured file exists" if key_ok else "configured file is missing",
        )
    )
    if key_ok and _bot_dependencies_available():
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
    results.extend(_state_results(settings.workspace_root, settings.database_path))
    results.extend(_catalog_results(settings.repositories_config))
    results.append(
        BotDoctorResult(
            status="PASS",
            name="docker_separation",
            message="control configuration does not require Docker access",
        )
    )
    return results


async def run_bot_worker_doctor(
    settings: BotWorkerSettings,
    model_settings: Settings,
) -> list[BotDoctorResult]:
    """检查 Worker、模型配置和本机不可变镜像。"""

    results = _base_results(("git", "docker", "bash", "rg"))
    for name, configured in (
        ("MODEL_ID", bool(model_settings.model_id)),
        ("ANTHROPIC_API_KEY", bool(model_settings.anthropic_api_key)),
    ):
        results.append(
            BotDoctorResult(
                status="PASS" if configured else "FAIL",
                name=name,
                message="configured" if configured else "missing",
            )
        )
    results.extend(_state_results(settings.workspace_root, settings.database_path))
    try:
        catalog = load_repository_catalog(settings.repositories_config)
        if shutil.which("docker"):
            for name, config in catalog.repositories.items():
                image_id = await resolve_image_id(config.image)
                if image_id != config.image:
                    raise ValueError(f"local image mismatch for {name}")
                results.append(
                    BotDoctorResult(status="PASS", name=f"image:{name}", message=image_id)
                )
        results.append(
            BotDoctorResult(
                status="PASS",
                name="repositories",
                message=f"{len(catalog.repositories)} configured",
            )
        )
    except Exception as exc:
        results.append(
            BotDoctorResult(status="FAIL", name="repositories", message=str(exc)[:500])
        )
    return results


def _base_results(programs: Iterable[str]) -> list[BotDoctorResult]:
    results = [
        BotDoctorResult(
            status="PASS" if sys.platform == "linux" else "FAIL",
            name="operating_system",
            message="Linux host" if sys.platform == "linux" else "bot services require Linux",
        )
    ]
    dependencies_ok = _bot_dependencies_available()
    results.append(
        BotDoctorResult(
            status="PASS" if dependencies_ok else "FAIL",
            name="bot_dependencies",
            message="available" if dependencies_ok else "install the project with .[bot]",
        )
    )
    for program in programs:
        path = shutil.which(program)
        results.append(
            BotDoctorResult(
                status="PASS" if path else "FAIL",
                name=program,
                message="available" if path else "not found on PATH",
            )
        )
    return results


def _bot_dependencies_available() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
        import jwt  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        return False
    return True


def _state_results(workspace_root: Path, database_path: Path) -> list[BotDoctorResult]:
    results: list[BotDoctorResult] = []
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        probe = workspace_root / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        results.append(BotDoctorResult(status="PASS", name="workspace", message="writable"))
    except OSError:
        results.append(BotDoctorResult(status="FAIL", name="workspace", message="not writable"))
    try:
        BotStore(database_path).initialize()
        results.append(
            BotDoctorResult(status="PASS", name="sqlite", message="WAL schema initialized")
        )
    except Exception:
        results.append(
            BotDoctorResult(status="FAIL", name="sqlite", message="database initialization failed")
        )
    return results


def _catalog_results(path: Path) -> list[BotDoctorResult]:
    try:
        catalog = load_repository_catalog(path)
        return [
            BotDoctorResult(
                status="PASS",
                name="repositories",
                message=f"{len(catalog.repositories)} configured with immutable image IDs",
            )
        ]
    except Exception as exc:
        return [BotDoctorResult(status="FAIL", name="repositories", message=str(exc)[:500])]
