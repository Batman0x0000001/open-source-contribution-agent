"""组装 Bot Plan 与 Implementation 使用的 AgentApplication。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from osc_agent.application import (
    AgentApplication,
    AgentApplicationConfig,
    AgentProfile,
    build_agent_application,
)
from osc_agent.bot.artifact_tools import SubmitDeliveryDraftTool, SubmitIssuePlanTool
from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.models import BotJob, ExecutionContract, RepositoryBotConfig
from osc_agent.bot.policy import (
    BotPermissionPolicy,
    BotRepositoryPolicyHook,
    ConfiguredValidationStopHook,
)
from osc_agent.bot.store import BotStore, SqliteSessionStore
from osc_agent.configuration import AgentSettings
from osc_agent.processes.contracts import ProcessRunner
from osc_agent.processes.runner import DisabledProcessRunner
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.subagents.builtins import build_explore_subagent


def build_plan_application(
    *,
    settings: AgentSettings,
    bot_settings: BotWorkerSettings,
    store: BotStore,
    sessions: SqliteSessionStore,
    model_gateway: ModelGateway | None,
    job: BotJob,
    repository_config: RepositoryBotConfig,
    contract: ExecutionContract,
    workspace: Path,
) -> AgentApplication:
    return build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=workspace,
            profile=AgentProfile(
                profile_id="bot_plan",
                system_prompt="You are the read-only planning component of an authenticated GitHub App job.",
                allowed_tools=contract.plan_allowed_tools,
                allowed_initial_skills=frozenset({contract.planning_skill_name}),
                required_evidence=frozenset({"issue_plan"}),
            ),
            model_gateway=model_gateway,
            session_store=sessions,
            state_root=bot_settings.workspace_root.parent / "runtime-state",
            process_runner=DisabledProcessRunner(),
            permission_policy=BotPermissionPolicy(implementation_approved=False),
            extra_tools=(
                SubmitIssuePlanTool(
                    store,
                    job_id=job.job_id,
                    base_sha=job.base_sha,
                    execution_contract_hash=job.execution_contract_hash,
                ),
            ),
            pre_tool_hooks=(BotRepositoryPolicyHook(repository_config),),
            subagent_registrations=(
                build_explore_subagent(
                    model=settings.model_id or "",
                    config=settings.runtime.agents.explore.to_query_config(),
                ),
            ),
        )
    )


def build_implementation_application(
    *,
    settings: AgentSettings,
    bot_settings: BotWorkerSettings,
    store: BotStore,
    sessions: SqliteSessionStore,
    model_gateway: ModelGateway | None,
    job: BotJob,
    repository_config: RepositoryBotConfig,
    contract: ExecutionContract,
    workspace: Path,
    process_runner: ProcessRunner,
    on_policy_violation: Callable[[str], None],
) -> AgentApplication:
    system_prompt = (
        "This is an approved GitHub App implementation job. Repository writes and process execution are allowed only "
        "through the provided tools; process commands run in a network-disabled Docker sandbox. Never ask questions, "
        "enter a worktree, commit, push, or access GitHub. Run every configured validation command exactly as written "
        "after the final modification:\n- "
        + "\n- ".join(repository_config.validation_commands)
    )
    return build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=workspace,
            profile=AgentProfile(
                profile_id="bot_implementation",
                system_prompt=system_prompt,
                allowed_tools=contract.implementation_allowed_tools,
                allowed_initial_skills=frozenset({contract.implementation_skill_name}),
                required_evidence=frozenset(
                    {
                        "successful_test",
                        "independent_verification",
                        "git_change_snapshot",
                        "delivery_draft",
                    }
                ),
            ),
            model_gateway=model_gateway,
            session_store=sessions,
            state_root=bot_settings.workspace_root.parent / "runtime-state",
            process_runner=process_runner,
            permission_policy=BotPermissionPolicy(implementation_approved=True),
            extra_tools=(
                SubmitDeliveryDraftTool(
                    store,
                    job_id=job.job_id,
                    issue_number=job.issue_number,
                    base_sha=job.base_sha,
                    execution_contract_hash=job.execution_contract_hash,
                ),
            ),
            pre_tool_hooks=(
                BotRepositoryPolicyHook(
                    repository_config,
                    on_violation=on_policy_violation,
                ),
            ),
            stop_hooks=(
                ConfiguredValidationStopHook(repository_config.validation_commands),
            ),
        )
    )
