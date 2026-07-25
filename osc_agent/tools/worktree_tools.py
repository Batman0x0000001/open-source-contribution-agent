from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import Field

from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.models import ContextUpdate, ContractModel, ToolResult, ToolUseContext, ValidationFailure, ValidationResult, ValidationSuccess
from osc_agent.runtime.tool import BaseTool


class EnterWorktreeInput(ContractModel):
    name: str = Field(min_length=1, max_length=64)


class WorktreeOutput(ContractModel):
    path: str
    branch: str
    action: Literal["entered", "kept", "removed"]


class EnterWorktreeTool(BaseTool[EnterWorktreeInput, WorktreeOutput]):
    name = "enter_worktree"
    description = "Create and enter a Git worktree isolated for the current session."
    input_model = EnterWorktreeInput
    output_model = WorktreeOutput

    def __init__(self, manager: WorktreeManager) -> None:
        self.manager = manager

    def is_destructive(self, input: EnterWorktreeInput) -> bool:
        return True

    async def validate_input(self, input: EnterWorktreeInput, context: ToolUseContext) -> ValidationResult:
        if context.worktree is not None:
            return ValidationFailure(reason="session is already inside a worktree")
        try:
            self.manager.validate_name(input.name)
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        return ValidationSuccess()

    async def call(self, input: EnterWorktreeInput, context: ToolUseContext) -> ToolResult:
        session = await asyncio.to_thread(self.manager.create, Path(context.working_directory), input.name)
        return ToolResult(
            data={"path": session.path, "branch": session.branch, "action": "entered"},
            context_update=ContextUpdate(working_directory=session.path, worktree=session),
        )


class ExitWorktreeInput(ContractModel):
    action: Literal["keep", "remove", "discard"] = "keep"


class ExitWorktreeTool(BaseTool[ExitWorktreeInput, WorktreeOutput]):
    name = "exit_worktree"
    description = "Keep, safely remove, or explicitly discard the current session worktree."
    input_model = ExitWorktreeInput
    output_model = WorktreeOutput

    def __init__(self, manager: WorktreeManager) -> None:
        self.manager = manager

    def is_destructive(self, input: ExitWorktreeInput) -> bool:
        return input.action in {"remove", "discard"}

    async def validate_input(self, input: ExitWorktreeInput, context: ToolUseContext) -> ValidationResult:
        if context.worktree is None:
            return ValidationFailure(reason="session is not inside a worktree")
        return ValidationSuccess()

    async def call(self, input: ExitWorktreeInput, context: ToolUseContext) -> ToolResult:
        session = context.worktree
        assert session is not None
        if input.action in {"remove", "discard"}:
            await asyncio.to_thread(self.manager.remove, session, discard=input.action == "discard")
        return ToolResult(
            data={"path": session.path, "branch": session.branch, "action": "removed" if input.action != "keep" else "kept"},
            context_update=ContextUpdate(
                working_directory=session.original_working_directory,
                clear_worktree=True,
            ),
        )
