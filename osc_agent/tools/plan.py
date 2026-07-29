"""提供规划模式和会话计划文档 Tool。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from osc_agent.contracts import ContractModel
from osc_agent.runtime.state import (
    PlanModeEntered,
    PlanModeExited,
    PlanSaved,
    ToolContext,
)
from osc_agent.runtime.tool_models import (
    ToolResult,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool


class EmptyInput(ContractModel):
    pass


class ModeOutput(ContractModel):
    mode: str
    message: str
    plan_path: str | None = None


class EnterPlanModeTool(BaseTool[EmptyInput, ModeOutput]):
    name = "enter_plan_mode"
    description = "Request approval to enter read-only planning mode for the current session."
    input_model = EmptyInput
    output_model = ModeOutput

    def is_read_only(self, input: EmptyInput) -> bool:
        return True

    def is_destructive(self, input: EmptyInput) -> bool:
        return True

    async def validate_input(
        self, input: EmptyInput, context: ToolContext
    ) -> ValidationResult:
        if context.permissions.mode == "plan":
            return ValidationFailure(reason="session is already in plan mode")
        return ValidationSuccess()

    async def call(self, input: EmptyInput, context: ToolContext) -> ToolResult:
        return ToolResult(
            data={
                "mode": "plan",
                "message": "Entered plan mode",
                "plan_path": context.permissions.plan_path,
            },
            state_changes=(PlanModeEntered(),),
        )


class WritePlanInput(ContractModel):
    content: str = Field(min_length=1)


class WritePlanOutput(ContractModel):
    path: str
    chars_written: int = Field(ge=1)


class WritePlanTool(BaseTool[WritePlanInput, WritePlanOutput]):
    name = "write_plan"
    description = "Write the current session plan to its fixed user-state location while in plan mode."
    input_model = WritePlanInput
    output_model = WritePlanOutput

    def is_read_only(self, input: WritePlanInput) -> bool:
        # 这是 Plan Mode 唯一允许的受控写操作，不接受模型提供的路径。
        return False

    async def validate_input(
        self, input: WritePlanInput, context: ToolContext
    ) -> ValidationResult:
        if context.permissions.mode != "plan":
            return ValidationFailure(reason="write_plan is only available in plan mode")
        return ValidationSuccess()

    async def call(self, input: WritePlanInput, context: ToolContext) -> ToolResult:
        root = Path(context.state_directory) / "plans"
        path = root / f"{_safe_session_id(context.session_id)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(input.content, encoding="utf-8")
        return ToolResult(
            data={"path": path.name, "chars_written": len(input.content)},
            state_changes=(PlanSaved(path=path.name),),
        )


class ReadPlanOutput(ContractModel):
    path: str
    content: str


class ReadPlanTool(BaseTool[EmptyInput, ReadPlanOutput]):
    name = "read_plan"
    description = "Read the plan owned by the current session."
    input_model = EmptyInput
    output_model = ReadPlanOutput

    def is_read_only(self, input: EmptyInput) -> bool:
        return True

    async def validate_input(
        self, input: EmptyInput, context: ToolContext
    ) -> ValidationResult:
        if not context.permissions.plan_path:
            return ValidationFailure(reason="this session has no saved plan")
        try:
            path = _plan_path(context)
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        if not path.is_file():
            return ValidationFailure(reason="saved plan file does not exist")
        return ValidationSuccess()

    async def call(self, input: EmptyInput, context: ToolContext) -> ToolResult:
        path = _plan_path(context)
        return ToolResult(
            data={"path": path.name, "content": path.read_text(encoding="utf-8")}
        )


class ExitPlanModeTool(BaseTool[EmptyInput, ModeOutput]):
    name = "exit_plan_mode"
    description = "Request approval to leave plan mode after a session plan has been written."
    input_model = EmptyInput
    output_model = ModeOutput

    def is_read_only(self, input: EmptyInput) -> bool:
        return True

    def is_destructive(self, input: EmptyInput) -> bool:
        return True

    async def validate_input(
        self, input: EmptyInput, context: ToolContext
    ) -> ValidationResult:
        if context.permissions.mode != "plan":
            return ValidationFailure(reason="session is not in plan mode")
        if not context.permissions.plan_path:
            return ValidationFailure(reason="write a plan before exiting plan mode")
        try:
            path = _plan_path(context)
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        if not path.is_file():
            return ValidationFailure(reason="saved plan file does not exist")
        return ValidationSuccess()

    async def call(self, input: EmptyInput, context: ToolContext) -> ToolResult:
        return ToolResult(
            data={
                "mode": "default",
                "message": "Plan approved",
                "plan_path": context.permissions.plan_path,
            },
            state_changes=(PlanModeExited(),),
        )


def _plan_path(context: ToolContext) -> Path:
    if not context.permissions.plan_path:
        raise ValueError("session has no plan")
    root = (Path(context.state_directory) / "plans").resolve()
    expected_name = f"{_safe_session_id(context.session_id)}.md"
    if context.permissions.plan_path != expected_name:
        raise ValueError("plan does not belong to the current session")
    path = (root / context.permissions.plan_path).resolve()
    if path.parent != root:
        raise ValueError("invalid plan path")
    return path


def _safe_session_id(value: str) -> str:
    if not value or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in value
    ):
        raise ValueError("invalid session id")
    return value
