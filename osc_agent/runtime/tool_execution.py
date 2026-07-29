"""执行工具输入校验、权限检查、Hook 和输出校验流水线。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import JsonValue, ValidationError

from osc_agent.runtime.hooks import HookRegistry, PostToolUsePayload, PreToolUsePayload
from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.tool_models import (
    Allow,
    ApprovalResponse,
    Ask,
    Deny,
    ToolError,
    ToolResult,
    ValidationFailure,
)
from osc_agent.runtime.state import PermissionGrant, PermissionGranted, ToolContext
from osc_agent.runtime.permissions import DefaultPermissionPolicy, PermissionPolicy
from osc_agent.runtime.tool import Tool, ToolRegistry


ApprovalHandler = Callable[[Ask], Awaitable[ApprovalResponse]]
@dataclass(frozen=True)
class ToolExecutionDependencies:
    approval_handler: ApprovalHandler | None = None


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

    async def execute(self, call: ToolUseBlock, context: ToolContext) -> ToolResult:
        granted: list[PermissionGranted] = []
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
        generally_allowed_input = await self._resolve_permission(
            permission,
            parsed,
            tool,
            context,
            granted,
        )
        if isinstance(generally_allowed_input, ToolResult):
            return _with_grants(generally_allowed_input, granted)

        tool_permission = await tool.check_permissions(generally_allowed_input, context)
        allowed_input = await self._resolve_permission(
            tool_permission,
            generally_allowed_input,
            tool,
            context,
            granted,
        )
        if isinstance(allowed_input, ToolResult):
            return _with_grants(allowed_input, granted)

        serialized_input: dict[str, JsonValue] = allowed_input.model_dump(mode="json")
        hook_result = await self.hooks.run_pre_tool_use(
            PreToolUsePayload(tool_name=tool.name, input=serialized_input),
            context,
        )
        if not hook_result.allowed:
            return _with_grants(_error("HOOK_BLOCKED", hook_result.reason), granted)

        try:
            result = await tool.call(allowed_input, context)
        except Exception as exc:  # noqa: BLE001 - Tool 异常必须转换为结构化结果。
            result = _error("TOOL_EXECUTION_FAILED", str(exc) or type(exc).__name__)

        result = _validate_output(tool, result)
        result = _with_grants(result, granted)
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
        context: ToolContext,
        granted: list[PermissionGranted],
    ) -> ContractModel | ToolResult:
        if isinstance(decision, Deny):
            return _error("PERMISSION_DENIED", decision.reason)
        if isinstance(decision, Ask):
            grant = _permission_grant(decision, input, context)
            if grant is not None and grant in context.permissions.grants:
                return input
            if self.dependencies.approval_handler is None:
                return _error("PERMISSION_REQUIRED", decision.prompt)
            response = await self.dependencies.approval_handler(decision)
            if not isinstance(response, ApprovalResponse):
                return _error(
                    "PERMISSION_RESPONSE_INVALID",
                    "approval handler must return ApprovalResponse",
                )
            if response.choice == "deny":
                return _error("PERMISSION_DENIED", decision.prompt)
            if response.choice == "allow_for_session":
                if grant is None:
                    return _error(
                        "PERMISSION_RESPONSE_INVALID",
                        "this permission risk cannot be remembered for the session",
                    )
                if grant not in context.permissions.grants and all(
                    change.grant != grant for change in granted
                ):
                    granted.append(PermissionGranted(grant=grant))
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


def _with_grants(result: ToolResult, granted: list[PermissionGranted]) -> ToolResult:
    if not granted:
        return result
    return result.model_copy(
        update={"state_changes": (*granted, *result.state_changes)}, deep=True
    )


def _permission_grant(
    decision: Ask,
    input: ContractModel,
    context: ToolContext,
) -> PermissionGrant | None:
    if decision.risk not in {"write", "process"}:
        return None
    working_directory = os.path.normcase(
        str(Path(context.workspace.working_directory).resolve())
    )
    canonical = json.dumps(
        input.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(
        (
            f"{decision.tool_name}\0{decision.risk}\0"
            f"{working_directory}\0{canonical}"
        ).encode("utf-8")
    ).hexdigest()
    return PermissionGrant(
        tool_name=decision.tool_name,
        risk=decision.risk,
        working_directory=working_directory,
        input_fingerprint=fingerprint,
    )
