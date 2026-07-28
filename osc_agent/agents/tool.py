from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import Field, JsonValue, ValidationError

from osc_agent.agents.definitions import AgentInvocation
from osc_agent.agents.registry import AgentRegistry
from osc_agent.agents.runner import AgentRunner
from osc_agent.runtime.models import (
    ContractModel,
    ToolError,
    ToolResult,
    ToolUseContext,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.tools.git import git_workspace_fingerprint


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

    def __init__(self, runner: AgentRunner, registry: AgentRegistry) -> None:
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

    async def call(self, input: AgentToolInput, context: ToolUseContext) -> ToolResult:
        registration = self.registry.get(input.agent)
        if registration is None:
            return _error("AGENT_NOT_FOUND", f"unknown agent: {input.agent}")
        try:
            arguments = registration.input_model.model_validate(
                input.arguments,
                context={"repository_root": context.working_directory},
            )
        except ValidationError as exc:
            return _error("AGENT_INPUT_INVALID", str(exc))

        semaphore = self._semaphores.setdefault(
            input.agent,
            asyncio.Semaphore(registration.max_parallel),
        )
        async with semaphore:
            before = None
            workspace_fingerprint = None
            if registration.read_only:
                before = await _fingerprint(context)
                if isinstance(before, ToolResult):
                    return before
            prompt = (
                f"Task:\n{input.task}\n\n"
                + registration.prompt_builder(arguments)
            )
            parent_messages = (
                context.transcript_messages
                if registration.definition.context_policy == "fork"
                else []
            )
            try:
                run = await self.runner.run(
                    AgentInvocation(
                        agent_name=input.agent,
                        prompt=prompt,
                        mode=(
                            "fork"
                            if registration.definition.context_policy == "fork"
                            else "inline"
                        ),
                        parent_session_id=context.session_id,
                        working_directory=context.working_directory,
                        caller_capabilities=context.capabilities,
                        parent_messages=parent_messages,
                    )
                )
            except asyncio.CancelledError:
                if registration.read_only:
                    guard_error, workspace_fingerprint = await _guard_result(before, context)
                    if guard_error is not None:
                        return guard_error
                raise
            except Exception:
                if registration.read_only:
                    guard_error, workspace_fingerprint = await _guard_result(before, context)
                    if guard_error is not None:
                        return guard_error
                raise
            if registration.read_only:
                guard_error, workspace_fingerprint = await _guard_result(before, context)
                if guard_error is not None:
                    return guard_error
        if run.status != "completed":
            return ToolResult(
                data={
                    "agent": input.agent,
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
                context={"repository_root": context.working_directory},
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            return _error("AGENT_OUTPUT_INVALID", str(exc))
        return ToolResult(
            data={
                "agent": input.agent,
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


async def _fingerprint(context: ToolUseContext) -> str | ToolResult:
    try:
        return await asyncio.to_thread(
            git_workspace_fingerprint,
            repo_root=Path(context.working_directory),
        )
    except (OSError, ValueError) as exc:
        return _error(
            "AGENT_READ_ONLY_GUARD_FAILED",
            f"unable to verify read-only Agent workspace: {exc}",
        )


async def _guard_result(
    before: str | ToolResult | None,
    context: ToolUseContext,
) -> tuple[ToolResult | None, str | None]:
    after = await _fingerprint(context)
    if isinstance(after, ToolResult):
        return after, None
    if before != after:
        return (
            _error(
                "AGENT_READ_ONLY_VIOLATION",
                "read-only Agent changed Git-visible repository state; changes were preserved for inspection",
            ),
            after,
        )
    return None, after
