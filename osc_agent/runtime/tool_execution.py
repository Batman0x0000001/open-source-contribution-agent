from __future__ import annotations

from dataclasses import dataclass
import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import JsonValue, ValidationError

from osc_agent.runtime.hooks import HookRegistry, PostToolUsePayload, PreToolUsePayload
from osc_agent.runtime.models import (
    Allow,
    Ask,
    ContractModel,
    Deny,
    ToolError,
    ToolResult,
    ToolUseBlock,
    ToolUseContext,
    ValidationFailure,
)
from osc_agent.runtime.permissions import DefaultPermissionPolicy, PermissionPolicy
from osc_agent.runtime.tool import Tool, ToolRegistry
from osc_agent.tools.git import git_workspace_fingerprint


ApprovalHandler = Callable[[Ask], Awaitable[bool]]
QuestionHandler = Callable[[list[dict[str, JsonValue]]], Awaitable[dict[str, str]]]


@dataclass(frozen=True)
class ToolExecutionDependencies:
    approval_handler: ApprovalHandler | None = None
    question_handler: QuestionHandler | None = None


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_policy: PermissionPolicy | None = None,
        hooks: HookRegistry | None = None,
        dependencies: ToolExecutionDependencies | None = None,
    ) -> None:
        self.registry = registry
        self.permission_policy = permission_policy or DefaultPermissionPolicy()
        self.hooks = hooks or HookRegistry()
        self.dependencies = dependencies or ToolExecutionDependencies()

    async def execute(self, call: ToolUseBlock, context: ToolUseContext) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None:
            return _error("TOOL_NOT_FOUND", f"unknown or disabled tool: {call.name}")

        parsed = _parse_input(tool, call.input)
        if isinstance(parsed, ToolResult):
            return parsed

        validation = await tool.validate_input(parsed, context)
        if isinstance(validation, ValidationFailure):
            return _error("TOOL_VALIDATION_FAILED", validation.reason)

        permission = await self.permission_policy.decide(tool, parsed, context)
        generally_allowed_input = await self._resolve_permission(permission, parsed, tool)
        if isinstance(generally_allowed_input, ToolResult):
            return generally_allowed_input

        tool_permission = await tool.check_permissions(generally_allowed_input, context)
        allowed_input = await self._resolve_permission(tool_permission, generally_allowed_input, tool)
        if isinstance(allowed_input, ToolResult):
            return allowed_input

        serialized_input: dict[str, JsonValue] = allowed_input.model_dump(mode="json")
        hook_result = await self.hooks.run_pre_tool_use(
            PreToolUsePayload(tool_name=tool.name, input=serialized_input),
            context,
        )
        if not hook_result.allowed:
            return _error("HOOK_BLOCKED", hook_result.reason)

        try:
            if tool.name == "ask_user_question":
                if self.dependencies.question_handler is None:
                    result = _error("USER_INTERACTION_REQUIRED", "no question handler is configured")
                else:
                    questions = serialized_input.get("questions")
                    if not isinstance(questions, list):
                        result = _error("TOOL_INPUT_INVALID", "questions must be a list")
                    else:
                        raw_answers = await self.dependencies.question_handler(questions)
                        answers = []
                        for question in questions:
                            if not isinstance(question, dict):
                                continue
                            question_id = str(question.get("id") or "")
                            text = str(question.get("question") or "")
                            raw = raw_answers.get(question_id, raw_answers.get(text))
                            if raw is None:
                                result = _error(
                                    "USER_INTERACTION_INVALID",
                                    f"no answer was returned for question {question_id or text}",
                                )
                                break
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
                        else:
                            try:
                                fingerprint = await asyncio.to_thread(
                                    git_workspace_fingerprint,
                                    repo_root=Path(context.working_directory),
                                )
                            except (OSError, ValueError):
                                fingerprint = None
                            result = ToolResult(
                                data={
                                    "questions": questions,
                                    "answers": answers,
                                    "workspace_fingerprint": fingerprint,
                                }
                            )
            else:
                result = await tool.call(allowed_input, context)
        except Exception as exc:  # noqa: BLE001 - Tool 异常必须转换为结构化结果。
            result = _error("TOOL_EXECUTION_FAILED", str(exc) or type(exc).__name__)

        result = _validate_output(tool, result)
        await self.hooks.run_post_tool_use(
            PostToolUsePayload(tool_name=tool.name, input=serialized_input, result=result),
            context,
        )
        return result

    async def _resolve_permission(
        self,
        decision: Allow | Deny | Ask,
        input: ContractModel,
        tool: Tool[ContractModel, ContractModel],
    ) -> ContractModel | ToolResult:
        if isinstance(decision, Deny):
            return _error("PERMISSION_DENIED", decision.reason)
        if isinstance(decision, Ask):
            if self.dependencies.approval_handler is None:
                return _error("PERMISSION_REQUIRED", decision.prompt)
            if not await self.dependencies.approval_handler(decision):
                return _error("PERMISSION_DENIED", decision.prompt)
            return input
        try:
            return tool.input_model.model_validate(decision.updated_input)
        except ValidationError as exc:
            return _error("PERMISSION_INPUT_INVALID", str(exc))


def _parse_input(
    tool: Tool[ContractModel, ContractModel],
    raw_input: dict[str, JsonValue],
) -> ContractModel | ToolResult:
    try:
        return tool.input_model.model_validate(raw_input)
    except ValidationError as exc:
        return _error("TOOL_INPUT_INVALID", str(exc))


def _validate_output(
    tool: Tool[ContractModel, ContractModel],
    result: ToolResult,
) -> ToolResult:
    if result.error is not None:
        return result
    try:
        output = tool.output_model.model_validate(result.data)
    except ValidationError as exc:
        return _error("TOOL_OUTPUT_INVALID", str(exc))
    return result.model_copy(update={"data": output.model_dump(mode="json")})


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(error=ToolError(code=code, message=message))
