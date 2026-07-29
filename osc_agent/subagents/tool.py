"""将子 Agent 调用封装为主 Runtime 可使用的 agent 工具。"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Literal

from pydantic import Field, JsonValue, ValidationError

from osc_agent.contracts import ContractModel
from osc_agent.runtime.tool_models import (
    ToolError,
    ToolResult,
)
from osc_agent.runtime.state import ToolContext
from osc_agent.runtime.tool import BaseTool
from osc_agent.subagents.models import SubagentRequest, SubagentRunResult
from osc_agent.subagents.registry import (
    SubagentContract,
    SubagentRegistration,
    SubagentRegistry,
)
from osc_agent.subagents.runner import SubagentRunner
from osc_agent.subagents.workspace_guard import (
    capture_workspace_fingerprint,
    verify_workspace_unchanged,
)


MAX_AGENT_OUTPUT_CHARS = 30_000


class AgentToolInput(ContractModel):
    agent: str = Field(min_length=1)
    task: str = Field(min_length=1, max_length=4_000)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class AgentToolOutput(ContractModel):
    agent: str
    status: Literal["completed", "failed", "cancelled"]
    child_session_id: str
    result: JsonValue = None
    error: str | None = None
    workspace_fingerprint: str | None = None


class AgentTool(BaseTool[AgentToolInput, AgentToolOutput]):
    name = "agent"
    input_model = AgentToolInput
    output_model = AgentToolOutput

    def __init__(self, runner: SubagentRunner, registry: SubagentRegistry) -> None:
        self.runner = runner
        self.registry = registry
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    @property
    def description(self) -> str:
        available = []
        for registration in self.registry.list():
            schema = json.dumps(
                registration.input_model.model_json_schema(mode="validation"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            available.append(
                f"{registration.definition.name}: {registration.definition.description}; "
                f"arguments schema={schema}"
            )
        return (
            "Run one code-registered child Agent with a bounded task and validated arguments. "
            f"Available: {'; '.join(available) if available else 'none'}"
        )

    def is_read_only(self, input: AgentToolInput) -> bool:
        registration = self.registry.get(input.agent)
        return bool(registration and registration.read_only)

    def is_concurrency_safe(self, input: AgentToolInput) -> bool:
        registration = self.registry.get(input.agent)
        return bool(registration and registration.concurrency_safe)

    def is_destructive(self, input: AgentToolInput) -> bool:
        registration = self.registry.get(input.agent)
        return bool(registration and not registration.read_only)

    async def call(self, input: AgentToolInput, context: ToolContext) -> ToolResult:
        registration = self.registry.get(input.agent)
        if registration is None:
            return _error("AGENT_NOT_FOUND", f"unknown agent: {input.agent}")

        arguments = self._validate_arguments(registration, input, context)
        if isinstance(arguments, ToolResult):
            return arguments

        prompt = f"Task:\n{input.task}\n\n" + registration.prompt_builder(arguments)
        executed, workspace_fingerprint = await self._execute(
            input.agent,
            registration,
            prompt,
            context,
        )
        if isinstance(executed, ToolResult):
            return executed
        return self._validate_result(
            input.agent,
            registration,
            executed,
            workspace_fingerprint,
            context,
        )

    @staticmethod
    def _validate_arguments(
        registration: SubagentRegistration,
        input: AgentToolInput,
        context: ToolContext,
    ) -> SubagentContract | ToolResult:
        try:
            return registration.input_model.model_validate(
                input.arguments,
                context={"repository_root": context.workspace.working_directory},
            )
        except ValidationError as exc:
            return _error("AGENT_INPUT_INVALID", str(exc))

    async def _execute(
        self,
        name: str,
        registration: SubagentRegistration,
        prompt: str,
        context: ToolContext,
    ) -> tuple[SubagentRunResult | ToolResult, str | None]:
        semaphore = self._semaphores.setdefault(
            name,
            asyncio.Semaphore(registration.max_parallel),
        )
        async with semaphore:
            before = None
            workspace_fingerprint = None
            if registration.read_only:
                before = await capture_workspace_fingerprint(context)
                if isinstance(before, ToolResult):
                    return before, None
            try:
                run = await self.runner.run(
                    name,
                    SubagentRequest(
                        prompt=prompt,
                        working_directory=context.workspace.working_directory,
                        caller_capabilities=context.capabilities,
                        parent_messages=tuple(context.transcript_messages),
                    ),
                )
            except (asyncio.CancelledError, Exception):
                if registration.read_only:
                    guard_error, workspace_fingerprint = await verify_workspace_unchanged(
                        before,
                        context,
                    )
                    if guard_error is not None:
                        return guard_error, workspace_fingerprint
                raise
            if registration.read_only:
                guard_error, workspace_fingerprint = await verify_workspace_unchanged(
                    before,
                    context,
                )
                if guard_error is not None:
                    return guard_error, workspace_fingerprint
        return run, workspace_fingerprint

    @staticmethod
    def _validate_result(
        name: str,
        registration: SubagentRegistration,
        run: SubagentRunResult,
        workspace_fingerprint: str | None,
        context: ToolContext,
    ) -> ToolResult:
        if run.status != "completed":
            return ToolResult(
                data={
                    "agent": name,
                    "status": run.status,
                    "child_session_id": run.session_id,
                    "result": None,
                    "error": run.error or run.status,
                    "workspace_fingerprint": workspace_fingerprint,
                }
            )
        if len(run.output) > MAX_AGENT_OUTPUT_CHARS:
            return _error("AGENT_OUTPUT_INVALID", "agent output exceeds the 30000 character limit")
        try:
            raw_output = _parse_agent_output(run.output)
            output = registration.output_model.model_validate(
                raw_output,
                context={"repository_root": context.workspace.working_directory},
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            return _error("AGENT_OUTPUT_INVALID", str(exc))
        return ToolResult(
            data={
                "agent": name,
                "status": "completed",
                "child_session_id": run.session_id,
                "result": output.model_dump(mode="json"),
                "error": None,
                "workspace_fingerprint": workspace_fingerprint,
            }
        )


def _error(code: str, message: str) -> ToolResult:
    return ToolResult(error=ToolError(code=code, message=message))


def _parse_agent_output(output: str) -> JsonValue:
    try:
        return json.loads(output)
    except json.JSONDecodeError as direct_error:
        fenced = re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if len(fenced) != 1:
            raise direct_error
        return json.loads(fenced[0])
