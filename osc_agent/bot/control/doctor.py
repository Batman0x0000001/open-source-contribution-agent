"""检查 Bot Control 的部署依赖与凭据边界。"""

from __future__ import annotations

import stat
import sys

from osc_agent.bot.config import BotControlSettings
from osc_agent.bot.diagnostics import (
    BotDoctorResult,
    base_results,
    bot_dependencies_available,
    catalog_results,
    state_results,
)


async def run_bot_control_doctor(settings: BotControlSettings) -> list[BotDoctorResult]:
    results = base_results(("git",))
    key = settings.github_app_private_key_path
    key_ok = key.is_file()
    results.append(
        BotDoctorResult(
            status="PASS" if key_ok else "FAIL",
            name="github_app_private_key",
            message="configured file exists" if key_ok else "configured file is missing",
        )
    )
    if key_ok and bot_dependencies_available():
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
    results.extend(state_results(settings.workspace_root, settings.database_path))
    results.extend(catalog_results(settings.repositories_config))
    results.append(
        BotDoctorResult(
            status="PASS",
            name="docker_separation",
            message="control configuration does not require Docker access",
        )
    )
    return results
