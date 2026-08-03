"""检查 Bot Worker 的模型、Docker 与不可变镜像。"""

from __future__ import annotations

import shutil

from osc_agent.bot.config import BotWorkerSettings, load_repository_catalog
from osc_agent.bot.diagnostics import BotDoctorResult, base_results, state_results
from osc_agent.bot.worker.docker_runner import resolve_image_id
from osc_agent.configuration import AgentSettings


async def run_bot_worker_doctor(
    settings: BotWorkerSettings,
    model_settings: AgentSettings,
) -> list[BotDoctorResult]:
    results = base_results(("git", "docker", "bash", "rg"))
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
    results.extend(state_results(settings.workspace_root, settings.database_path))
    try:
        catalog = load_repository_catalog(settings.repositories_config)
        if shutil.which("docker"):
            for name, config in catalog.repositories.items():
                image_id = await resolve_image_id(config.image)
                if image_id != config.image:
                    raise ValueError(f"local image mismatch for {name}")
                results.append(BotDoctorResult(status="PASS", name=f"image:{name}", message=image_id))
        results.append(
            BotDoctorResult(
                status="PASS",
                name="repositories",
                message=f"{len(catalog.repositories)} configured",
            )
        )
    except Exception as exc:
        results.append(BotDoctorResult(status="FAIL", name="repositories", message=str(exc)[:500]))
    return results
