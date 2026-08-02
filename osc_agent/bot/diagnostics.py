"""提供 Control 与 Worker Doctor 共用的无进程归属诊断。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import shutil
import sys

from osc_agent.bot.config import load_repository_catalog
from osc_agent.bot.persistence.store import BotStore
from osc_agent.contracts import FrozenContractModel


class BotDoctorResult(FrozenContractModel):
    status: str
    name: str
    message: str


def base_results(programs: Iterable[str]) -> list[BotDoctorResult]:
    results = [
        BotDoctorResult(
            status="PASS" if sys.platform == "linux" else "FAIL",
            name="operating_system",
            message="Linux host" if sys.platform == "linux" else "bot services require Linux",
        )
    ]
    dependencies_ok = bot_dependencies_available()
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


def bot_dependencies_available() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx2  # noqa: F401
        import jwt  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        return False
    return True


def state_results(workspace_root: Path, database_path: Path) -> list[BotDoctorResult]:
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
        results.append(BotDoctorResult(status="PASS", name="sqlite", message="WAL schema initialized"))
    except Exception:
        results.append(BotDoctorResult(status="FAIL", name="sqlite", message="database initialization failed"))
    return results


def catalog_results(path: Path) -> list[BotDoctorResult]:
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
