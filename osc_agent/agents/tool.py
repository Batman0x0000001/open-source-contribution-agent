from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from osc_agent.agents.definitions import AgentInvocation
from osc_agent.agents.runner import AgentRunner
from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.models import ContractModel, ToolResult, ToolUseContext, ValidationFailure, ValidationResult, ValidationSuccess
from osc_agent.runtime.tool import BaseTool


class AgentToolInput(ContractModel):
    agent: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    mode: Literal["inline", "fork"] = "inline"
    isolation: Literal["worktree"] | None = None


class AgentToolOutput(ContractModel):
    task_id: str
    session_id: str
    status: Literal["completed", "failed", "cancelled"]
    output: str = ""
    error: str | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None


class AgentTool(BaseTool[AgentToolInput, AgentToolOutput]):
    name = "agent"
    input_model = AgentToolInput
    output_model = AgentToolOutput

    def __init__(self, runner: AgentRunner, manager: WorktreeManager) -> None:
        self.runner = runner
        self.manager = manager

    @property
    def description(self) -> str:
        available = "; ".join(
            f"{definition.name}: {definition.description}"
            for definition in self.runner.list_definitions()
        )
        return f"Run a child Agent inline or with a forked transcript. Available: {available or 'none'}"

    async def validate_input(self, input: AgentToolInput, context: ToolUseContext) -> ValidationResult:
        return ValidationSuccess()

    async def call(self, input: AgentToolInput, context: ToolUseContext) -> ToolResult:
        worktree = None
        working_directory = context.working_directory
        if input.isolation == "worktree":
            name = f"agent-{uuid4().hex[:12]}"
            worktree = await asyncio.to_thread(
                self.manager.create,
                Path(context.working_directory),
                name,
            )
            working_directory = worktree.path
        result = await self.runner.run(
            AgentInvocation(
                agent_name=input.agent,
                prompt=input.prompt,
                mode=input.mode,
                parent_session_id=context.session_id,
                working_directory=working_directory,
                caller_capabilities=context.capabilities,
                parent_messages=context.transcript_messages if input.mode == "fork" else [],
            )
        )
        data = result.model_dump(mode="json")
        if worktree is not None:
            if await asyncio.to_thread(self.manager.is_dirty, worktree) or await asyncio.to_thread(self.manager.commits_ahead, worktree) > 0:
                data.update({"worktree_path": worktree.path, "worktree_branch": worktree.branch})
            else:
                await asyncio.to_thread(self.manager.remove, worktree)
                data.update({"worktree_path": None, "worktree_branch": None})
        return ToolResult(data=data)
