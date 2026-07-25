from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import JsonValue

from osc_agent.agents.explore import build_explore_registration
from osc_agent.agents.verify import build_verify_registration
from osc_agent.agents.registry import AgentRegistry
from osc_agent.agents.runner import AgentRunner
from osc_agent.agents.tool import AgentTool
from osc_agent.config import Settings
from osc_agent.providers.anthropic import AnthropicModelGateway
from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.runtime.models import Ask, CapabilityScope, QueryConfig
from osc_agent.runtime.query import AgentRuntime
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor
from osc_agent.runtime.session_store import FileSessionStore
from osc_agent.runtime.session_store import FileToolResultStore
from osc_agent.runtime.context import ContextPipeline, GatewayContextSummarizer
from osc_agent.runtime.state_paths import ApplicationStatePaths
from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.instructions import RepositoryInstructionResolver
from osc_agent.runtime.hooks import HookRegistry
from osc_agent.runtime.completion import CompletionEvidenceStopHook
from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.executor import SkillExecutor
from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.tool import SkillTool
from osc_agent.skills.resource_tool import ReadSkillResourceTool
from osc_agent.skills.runner import SkillCommandRunner
from osc_agent.tools.core import build_core_tool_registry


ApprovalHandler = Callable[[Ask], Awaitable[bool]]
QuestionHandler = Callable[[list[dict[str, JsonValue]]], Awaitable[dict[str, str]]]


@dataclass(frozen=True)
class ApplicationServices:
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor
    runtime: AgentRuntime
    agent_registry: AgentRegistry
    agent_runner: AgentRunner
    skill_catalog: SkillCatalog
    skill_executor: SkillExecutor
    skill_command_runner: SkillCommandRunner
    query_config: QueryConfig
    general_capabilities: CapabilityScope
    discovery_prompt: str


def build_skill_catalog(repo_root: Path) -> SkillCatalog:
    builtin = Path(__file__).resolve().parent / "skills"
    return SkillCatalog(
        [
            SkillLoader(builtin, source="builtin"),
            SkillLoader(Path.home() / ".osc_agent" / "skills", source="user"),
            SkillLoader(repo_root.resolve() / ".osc_agent" / "skills", source="project"),
        ]
    )


def build_application(
    *,
    settings: Settings,
    repo_root: Path,
    approval_handler: ApprovalHandler | None = None,
    question_handler: QuestionHandler | None = None,
    model_gateway: ModelGateway | None = None,
) -> ApplicationServices:
    """唯一生产组装根：一个 Runtime、ToolExecutor、SkillExecutor 和权限链。"""

    if not settings.model_id:
        raise ValueError("MODEL_ID is required for model execution")
    model_id = settings.model_id
    query_config = QueryConfig(
        max_rounds=settings.max_agent_rounds,
        max_total_tokens=settings.max_total_tokens,
        deadline_seconds=max(settings.agent_deadline_seconds, 1),
        max_no_progress_rounds=settings.no_progress_limit,
    )
    state_paths = ApplicationStatePaths.for_repository(repo_root)
    tool_result_store = FileToolResultStore(state_paths.tool_results)
    worktree_manager = WorktreeManager(state_paths.worktrees)
    instruction_resolver = RepositoryInstructionResolver()
    catalog = build_skill_catalog(repo_root)
    registry = build_core_tool_registry(
        worktree_manager=worktree_manager,
        tool_result_store=tool_result_store,
        instruction_resolver=instruction_resolver,
    )
    registry.register(ReadSkillResourceTool(catalog))
    hooks = HookRegistry()
    hooks.register_stop(CompletionEvidenceStopHook())
    executor = ToolExecutor(
        registry,
        hooks=hooks,
        dependencies=ToolExecutionDependencies(
            approval_handler=approval_handler,
            question_handler=question_handler,
        ),
    )
    if model_gateway is None and not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when no ModelGateway is injected")
    gateway = model_gateway or AnthropicModelGateway.from_config(
        api_key=settings.anthropic_api_key or "",
        base_url=settings.anthropic_base_url,
    )
    runtime = AgentRuntime(
        QueryDependencies(
            model_gateway=gateway,
            tool_registry=registry,
            tool_executor=executor,
            session_store=FileSessionStore(state_paths.sessions),
            state_directory=str(state_paths.repository),
            worktree_manager=worktree_manager,
            instruction_resolver=instruction_resolver,
            context_pipeline=ContextPipeline(
                tool_result_store=tool_result_store,
                summarizer=GatewayContextSummarizer(gateway, model=model_id),
                instruction_resolver=instruction_resolver,
            ),
        )
    )
    agent_registry = AgentRegistry(
        [
            build_explore_registration(model=model_id),
            build_verify_registration(model=model_id),
        ]
    )
    runner = AgentRunner(runtime, agent_registry, default_model=model_id)
    skill_executor = SkillExecutor(catalog, agent_runner=runner, query_config=query_config)
    registry.register(SkillTool(skill_executor))
    registry.register(AgentTool(runner, agent_registry))
    general_capabilities = CapabilityScope(allowed_tools=frozenset(registry.names()))
    discovery_prompt = _discovery_prompt(
        catalog,
        agent_registry,
        general_capabilities,
    )
    skill_command_runner = SkillCommandRunner(
        skill_executor,
        runtime,
        model=model_id,
        config=query_config,
        system_prompt="Follow the invoked Skill instructions and use repository evidence.",
        discovery_prompt=lambda capabilities: _discovery_prompt(
            catalog,
            agent_registry,
            capabilities,
        ),
    )
    return ApplicationServices(
        tool_registry=registry,
        tool_executor=executor,
        runtime=runtime,
        agent_registry=agent_registry,
        agent_runner=runner,
        skill_catalog=catalog,
        skill_executor=skill_executor,
        skill_command_runner=skill_command_runner,
        query_config=query_config,
        general_capabilities=general_capabilities,
        discovery_prompt=discovery_prompt,
    )


def _discovery_prompt(
    catalog: SkillCatalog,
    agent_registry: AgentRegistry,
    capabilities: CapabilityScope,
) -> str:
    skills = [
        f"- {item.manifest.name}: {item.manifest.description}"
        for item in catalog.list()
        if not item.manifest.disable_model_invocation
    ]
    agents = (
        [
            f"- {item.definition.name}: {item.definition.description}"
            for item in agent_registry.list()
        ]
        if capabilities.permits_tool("agent")
        else []
    )
    return (
        "<available_skills>\n"
        + ("\n".join(skills) if skills else "(none)")
        + "\n</available_skills>\n<available_agents>\n"
        + ("\n".join(agents) if agents else "(none)")
        + "\n</available_agents>"
    )
