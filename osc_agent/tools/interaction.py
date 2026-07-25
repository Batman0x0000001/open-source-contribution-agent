from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from osc_agent.runtime.models import (
    ContextUpdate,
    ContractModel,
    ToolResult,
    ToolUseContext,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool


class QuestionOption(ContractModel):
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1)


class UserQuestion(ContractModel):
    header: str = Field(min_length=1, max_length=24)
    question: str = Field(min_length=1)
    options: list[QuestionOption] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def unique_options(self) -> "UserQuestion":
        labels = [option.label for option in self.options]
        if len(labels) != len(set(labels)):
            raise ValueError("question option labels must be unique")
        return self


class AskUserQuestionInput(ContractModel):
    questions: list[UserQuestion] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_questions(self) -> "AskUserQuestionInput":
        values = [item.question for item in self.questions]
        if len(values) != len(set(values)):
            raise ValueError("question texts must be unique")
        return self


class AskUserQuestionOutput(ContractModel):
    questions: list[UserQuestion]
    answers: dict[str, str]


class AskUserQuestionTool(BaseTool[AskUserQuestionInput, AskUserQuestionOutput]):
    name = "ask_user_question"
    description = "Ask the user one to four structured questions and return their answers."
    input_model = AskUserQuestionInput
    output_model = AskUserQuestionOutput

    def is_read_only(self, input: AskUserQuestionInput) -> bool:
        return True

    async def call(self, input: AskUserQuestionInput, context: ToolUseContext) -> ToolResult:
        raise RuntimeError("AskUserQuestion must be resolved by ToolExecutor")


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

    async def validate_input(self, input: EmptyInput, context: ToolUseContext) -> ValidationResult:
        if context.permission_mode == "plan":
            return ValidationFailure(reason="session is already in plan mode")
        return ValidationSuccess()

    async def call(self, input: EmptyInput, context: ToolUseContext) -> ToolResult:
        return ToolResult(
            data={"mode": "plan", "message": "Entered plan mode", "plan_path": context.plan_path},
            context_update=ContextUpdate(permission_mode="plan"),
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

    async def validate_input(self, input: WritePlanInput, context: ToolUseContext) -> ValidationResult:
        if context.permission_mode != "plan":
            return ValidationFailure(reason="write_plan is only available in plan mode")
        return ValidationSuccess()

    async def call(self, input: WritePlanInput, context: ToolUseContext) -> ToolResult:
        root = Path(context.state_directory) / "plans"
        path = root / f"{_safe_session_id(context.session_id)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(input.content, encoding="utf-8")
        return ToolResult(
            data={"path": path.name, "chars_written": len(input.content)},
            context_update=ContextUpdate(plan_path=path.name),
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

    async def validate_input(self, input: EmptyInput, context: ToolUseContext) -> ValidationResult:
        if not context.plan_path:
            return ValidationFailure(reason="this session has no saved plan")
        try:
            path = _plan_path(context)
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        if not path.is_file():
            return ValidationFailure(reason="saved plan file does not exist")
        return ValidationSuccess()

    async def call(self, input: EmptyInput, context: ToolUseContext) -> ToolResult:
        path = _plan_path(context)
        return ToolResult(data={"path": path.name, "content": path.read_text(encoding="utf-8")})


class ExitPlanModeTool(BaseTool[EmptyInput, ModeOutput]):
    name = "exit_plan_mode"
    description = "Request approval to leave plan mode after a session plan has been written."
    input_model = EmptyInput
    output_model = ModeOutput

    def is_read_only(self, input: EmptyInput) -> bool:
        return True

    def is_destructive(self, input: EmptyInput) -> bool:
        return True

    async def validate_input(self, input: EmptyInput, context: ToolUseContext) -> ValidationResult:
        if context.permission_mode != "plan":
            return ValidationFailure(reason="session is not in plan mode")
        if not context.plan_path:
            return ValidationFailure(reason="write a plan before exiting plan mode")
        try:
            path = _plan_path(context)
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        if not path.is_file():
            return ValidationFailure(reason="saved plan file does not exist")
        return ValidationSuccess()

    async def call(self, input: EmptyInput, context: ToolUseContext) -> ToolResult:
        return ToolResult(
            data={"mode": "default", "message": "Plan approved", "plan_path": context.plan_path},
            context_update=ContextUpdate(permission_mode="default"),
        )


def _plan_path(context: ToolUseContext) -> Path:
    if not context.plan_path:
        raise ValueError("session has no plan")
    root = (Path(context.state_directory) / "plans").resolve()
    expected_name = f"{_safe_session_id(context.session_id)}.md"
    if context.plan_path != expected_name:
        raise ValueError("plan does not belong to the current session")
    path = (root / context.plan_path).resolve()
    if path.parent != root:
        raise ValueError("invalid plan path")
    return path


def _safe_session_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value):
        raise ValueError("invalid session id")
    return value
