"""集中声明一次 Query 执行所需的运行时依赖。"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
from uuid import uuid4

from osc_agent.runtime.context import ContextPipeline
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.runtime.session_store import SessionStore
from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.instructions import RepositoryInstructionResolver


@dataclass(frozen=True)
class QueryDependencies:
    model_gateway: ModelGateway
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor
    monotonic: Callable[[], float] = perf_counter
    new_id: Callable[[], str] = lambda: str(uuid4())
    context_pipeline: ContextPipeline = field(default_factory=ContextPipeline)
    session_store: SessionStore | None = None
    state_directory: str | None = None
    worktree_manager: WorktreeManager | None = None
    instruction_resolver: RepositoryInstructionResolver = field(
        default_factory=RepositoryInstructionResolver
    )
