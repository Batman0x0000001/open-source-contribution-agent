"""提供结构化用户提问 Tool 及其模型可见契约。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from osc_agent.contracts import ContractModel
from osc_agent.runtime.tool_models import (
    ToolError,
    ToolResult,
    ToolUseContext,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.workspaces.git_state import git_workspace_fingerprint


QuestionHandler = Callable[
    [list[dict[str, JsonValue]]],
    Awaitable[dict[str, str]],
]


class QuestionOption(ContractModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1)


class RemoteMismatchReference(ContractModel):
    local_origin: str = Field(min_length=3, max_length=300)
    requested_repository: str = Field(min_length=3, max_length=300)


class UserQuestion(ContractModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    header: str = Field(min_length=1, max_length=24)
    question: str = Field(min_length=1)
    options: list[QuestionOption] = Field(min_length=2, max_length=4)
    purpose: Literal[
        "general",
        "verification_waiver",
        "independent_verification_waiver",
        "instruction_conflict",
        "remote_mismatch",
    ] = "general"
    reason: str | None = None
    alternative_checks: list[str] = Field(default_factory=list)
    unverified_checks: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    related_child_session_id: str | None = None
    remote_mismatch_reference: RemoteMismatchReference | None = None

    @model_validator(mode="after")
    def unique_options(self) -> "UserQuestion":
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError("question option ids must be unique")
        if self.purpose == "verification_waiver":
            if not self.reason or not self.alternative_checks or not self.risks:
                raise ValueError(
                    "verification waiver requires reason, alternative_checks, and risks"
                )
            required = {"proceed_without_tests", "do_not_proceed"}
            if not required <= set(ids):
                raise ValueError(
                    "verification waiver options must include proceed_without_tests and do_not_proceed"
                )
        if self.purpose == "independent_verification_waiver":
            if (
                not self.related_child_session_id
                or not self.reason
                or not self.alternative_checks
                or not self.unverified_checks
                or not self.risks
            ):
                raise ValueError(
                    "independent verification waiver requires a child session, reason, completed "
                    "checks, unverified checks, and risks"
                )
            required = {
                "proceed_with_partial_verification",
                "return_to_implementation",
            }
            if not required <= set(ids):
                raise ValueError(
                    "independent verification waiver options must include "
                    "proceed_with_partial_verification and return_to_implementation"
                )
        if self.purpose == "remote_mismatch":
            if self.remote_mismatch_reference is None:
                raise ValueError(
                    "remote mismatch confirmation requires the exact local and requested repositories"
                )
            if "proceed" not in set(ids):
                raise ValueError("remote mismatch options must include proceed")
        return self


class AskUserQuestionInput(ContractModel):
    questions: list[UserQuestion] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_questions(self) -> "AskUserQuestionInput":
        values = [item.id for item in self.questions]
        if len(values) != len(set(values)):
            raise ValueError("question ids must be unique")
        return self


class QuestionAnswer(ContractModel):
    question_id: str = Field(min_length=1)
    selected_option_id: str | None = None
    custom_text: str | None = None

    @model_validator(mode="after")
    def exactly_one_answer(self) -> "QuestionAnswer":
        if (self.selected_option_id is None) == (self.custom_text is None):
            raise ValueError("answer must contain one selected option or custom text")
        return self


class AskUserQuestionOutput(ContractModel):
    questions: list[UserQuestion]
    answers: list[QuestionAnswer]
    workspace_fingerprint: str | None = None


class AskUserQuestionTool(BaseTool[AskUserQuestionInput, AskUserQuestionOutput]):
    name = "ask_user_question"
    description = "Ask the user one to four structured questions and return their answers."
    input_model = AskUserQuestionInput
    output_model = AskUserQuestionOutput

    def __init__(self, question_handler: QuestionHandler | None = None) -> None:
        self.question_handler = question_handler

    def is_read_only(self, input: AskUserQuestionInput) -> bool:
        return True

    async def call(
        self,
        input: AskUserQuestionInput,
        context: ToolUseContext,
    ) -> ToolResult:
        if self.question_handler is None:
            return _error("USER_INTERACTION_REQUIRED", "no question handler is configured")

        questions = [question.model_dump(mode="json") for question in input.questions]
        raw_answers = await self.question_handler(questions)
        answers: list[dict[str, str | None]] = []
        for question in questions:
            question_id = str(question.get("id") or "")
            text = str(question.get("question") or "")
            raw = raw_answers.get(question_id, raw_answers.get(text))
            if raw is None:
                return _error(
                    "USER_INTERACTION_INVALID",
                    f"no answer was returned for question {question_id or text}",
                )
            options = question.get("options") or []
            selected = next(
                (
                    str(option.get("id"))
                    for option in options
                    if isinstance(option, dict)
                    and raw in {option.get("id"), option.get("label")}
                ),
                None,
            )
            answers.append(
                {
                    "question_id": question_id,
                    "selected_option_id": selected,
                    "custom_text": None if selected is not None else str(raw),
                }
            )

        try:
            fingerprint = await asyncio.to_thread(
                git_workspace_fingerprint,
                repo_root=Path(context.working_directory),
            )
        except (OSError, ValueError):
            fingerprint = None
        return ToolResult(
            data={
                "questions": questions,
                "answers": answers,
                "workspace_fingerprint": fingerprint,
            }
        )


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(error=ToolError(code=code, message=message))
