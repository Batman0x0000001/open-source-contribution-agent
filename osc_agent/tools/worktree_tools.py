"""提供进入和退出隔离 Git worktree 的工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import Field

from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.models import ContextUpdate, ContractModel, ToolResult, ToolUseContext, ValidationFailure, ValidationResult, ValidationSuccess
from osc_agent.runtime.tool import BaseTool
from osc_agent.runtime.instructions import RepositoryInstructionResolver


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

    def __init__(
        self,
        manager: WorktreeManager,
        instructions: RepositoryInstructionResolver | None = None,
    ) -> None:
        self.manager = manager
        self.instructions = instructions or RepositoryInstructionResolver()

    def is_destructive(self, input: EnterWorktreeInput) -> bool:
        return True

    def permission_risk(self, input: EnterWorktreeInput) -> str:
        return "write"

    def permission_preview(
        self,
        input: EnterWorktreeInput,
        context: ToolUseContext,
    ) -> dict[str, object]:
        return {
            "action": "create",
            "path": str((self.manager.worktrees_root / input.name).resolve()),
            "branch": f"osc-agent/{input.name}",
        }

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
        instruction_state = self.instructions.activate_root(Path(session.path))
        return ToolResult(
            data={"path": session.path, "branch": session.branch, "action": "entered"},
            context_update=ContextUpdate(
                working_directory=session.path,
                worktree=session,
                instruction_state=instruction_state,
                replace_instruction_state=True,
                replace_file_observations=True,
            ),
        )


class ExitWorktreeInput(ContractModel):
    action: Literal["keep", "remove", "discard"] = "keep"


class ExitWorktreeTool(BaseTool[ExitWorktreeInput, WorktreeOutput]):
    name = "exit_worktree"
    description = "Keep, safely remove, or explicitly discard the current session worktree."
    input_model = ExitWorktreeInput
    output_model = WorktreeOutput

    def __init__(
        self,
        manager: WorktreeManager,
        instructions: RepositoryInstructionResolver | None = None,
    ) -> None:
        self.manager = manager
        self.instructions = instructions or RepositoryInstructionResolver()

    def is_destructive(self, input: ExitWorktreeInput) -> bool:
        return input.action in {"remove", "discard"}

    def permission_risk(self, input: ExitWorktreeInput) -> str:
        return "destructive" if input.action == "discard" else "write"

    def permission_preview(
        self,
        input: ExitWorktreeInput,
        context: ToolUseContext,
    ) -> dict[str, object]:
        return {
            "action": input.action,
            "path": context.worktree.path if context.worktree else None,
            "branch": context.worktree.branch if context.worktree else None,
        }

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
                instruction_state=self.instructions.activate_root(
                    Path(session.original_working_directory)
                ),
                replace_instruction_state=True,
                replace_file_observations=True,
            ),
        )
