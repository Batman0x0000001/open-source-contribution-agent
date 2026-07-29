"""解析 Worker 执行前绑定的仓库、合同与工作区。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.domain.execution import ExecutionContract
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotCatalog, RepositoryBotConfig
from osc_agent.bot.persistence.store import BotStore
from osc_agent.configuration import AgentSettings


class BotModelContractMismatch(ValueError):
    """执行合同绑定的模型与 Worker 实际模型不一致。"""


@dataclass(frozen=True)
class ResolvedJobContext:
    repository: RepositoryBotConfig
    contract: ExecutionContract
    workspace: Path


def resolve_job_context(
    job: BotJob,
    *,
    settings: AgentSettings,
    bot_settings: BotWorkerSettings,
    catalog: RepositoryBotCatalog,
    store: BotStore,
) -> ResolvedJobContext:
    configured = catalog.repositories.get(job.repository_full_name)
    contract = store.get_execution_contract(job.execution_contract_hash)
    plan_phase = job.status == "running_plan"
    workspace_value = job.plan_workspace_path if plan_phase else job.implementation_workspace_path
    ready = job.plan_workspace_ready if plan_phase else job.implementation_workspace_ready
    if configured is None or not configured.enabled or contract is None or not workspace_value or not ready:
        raise ValueError("job has no enabled repository configuration or prepared workspace")
    if (
        contract.contract_hash != job.execution_contract_hash
        or contract.repository_id != job.repository_id
        or contract.repository_full_name != job.repository_full_name
        or contract.base_sha != job.base_sha
        or contract.issue_input_hash != job.issue_input_hash
    ):
        raise ValueError("Job does not match its execution contract")
    if not settings.model_id or contract.model_id != settings.model_id:
        raise BotModelContractMismatch(
            "execution contract model does not match the configured Worker model"
        )
    repository = type(configured)(
        enabled=True,
        image=contract.image_id,
        validation_commands=contract.validation_commands,
        denied_paths=contract.denied_paths,
        max_changed_files=contract.max_changed_files,
        max_patch_bytes=contract.max_patch_bytes,
        command_timeout_seconds=contract.command_timeout_seconds,
        container_cpus=contract.container_cpus,
        container_memory=contract.container_memory,
        container_pids=contract.container_pids,
        pull_request_mode=contract.pull_request_mode,
    )
    workspace = Path(workspace_value).resolve()
    if not workspace.is_relative_to(bot_settings.workspace_root.resolve()):
        raise ValueError("prepared workspace escapes the bot workspace root")
    return ResolvedJobContext(repository=repository, contract=contract, workspace=workspace)
