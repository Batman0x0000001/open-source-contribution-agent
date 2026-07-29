"""集中声明一次 Query 执行所需的运行时依赖。"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Protocol

from osc_agent.runtime.context import ContextPipeline
from osc_agent.runtime.gateway import ModelGateway
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.runtime.session_store import SessionStore
from osc_agent.workspaces.instructions import RepositoryInstructionResolver
from osc_agent.workspaces.models import WorktreeSession


class WorkspaceSessionValidator(Protocol):
    def validate_session(self, session: WorktreeSession) -> None: ...


@dataclass(frozen=True)
class QueryDependencies:
    model_gateway: ModelGateway
    tool_executor: ToolExecutor
    state_directory: str
    monotonic: Callable[[], float] = perf_counter
    context_pipeline: ContextPipeline = field(default_factory=ContextPipeline)
    session_store: SessionStore | None = None
    workspace_validator: WorkspaceSessionValidator | None = None
    instruction_resolver: RepositoryInstructionResolver = field(
        default_factory=RepositoryInstructionResolver
    )
