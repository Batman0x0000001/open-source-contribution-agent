"""验证会话权限的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import Field

from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import RuntimeMessage, TextBlock, ToolUseBlock
from osc_agent.runtime.query_models import ResumeQueryParams, StartQueryParams
from osc_agent.runtime.tool_models import ApprovalResponse, Ask, ToolResult
from osc_agent.runtime.query import AgentRuntime
from osc_agent.runtime.session_store import FileSessionStore
from osc_agent.runtime.tool import BaseTool, ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor


class MutateInput(ContractModel):
    value: str = Field(min_length=1)


class MutateOutput(ContractModel):
    value: str


class MutateTool(BaseTool[MutateInput, MutateOutput]):
    name = "mutate"
    input_model = MutateInput
    output_model = MutateOutput

    def is_destructive(self, input: MutateInput) -> bool:
        return True

    def permission_risk(self, input: MutateInput) -> str:
        return "process"

    async def call(self, input: MutateInput, context: tool_context) -> ToolResult:
        return ToolResult(data={"value": input.value})


class Gateway:
    def __init__(self, messages: list[RuntimeMessage]) -> None:
        self.messages = list(messages)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        message = self.messages.pop(0)
        yield ModelCompleted(
            message=message,
            stop_reason=(
                "tool_use"
                if any(isinstance(block, ToolUseBlock) for block in message.content)
                else "end_turn"
            ),
        )


def _runtime(gateway, store, approval_handler) -> AgentRuntime:
    registry = ToolRegistry([MutateTool()])
    return AgentRuntime(
        QueryDependencies(
            model_gateway=gateway,
            tool_executor=ToolExecutor(
                registry,
                dependencies=ToolExecutionDependencies(
                    approval_handler=approval_handler,
                ),
            ),
            state_directory=str(store.root.parent / "state"),
            session_store=store,
        )
    )


async def _collect(runtime: AgentRuntime, params):
    return [event async for event in runtime.query(params)]


def test_exact_permission_grant_persists_across_resume(tmp_path: Path) -> None:
    store = FileSessionStore(tmp_path / "sessions")
    call = RuntimeMessage(
        role="assistant",
        content=[ToolUseBlock(id="call-1", name="mutate", input={"value": "same"})],
    )
    final = RuntimeMessage(role="assistant", content=[TextBlock(text="done")])
    approvals = 0

    async def approve(_decision: Ask) -> ApprovalResponse:
        nonlocal approvals
        approvals += 1
        return ApprovalResponse(choice="allow_for_session")

    first = _runtime(Gateway([call, final]), store, approve)
    asyncio.run(
        _collect(
            first,
            StartQueryParams(
                session_id="session-1",
                model="test",
                workspace_root=str(tmp_path),
                messages=[
                    RuntimeMessage(
                        role="user",
                        content=[TextBlock(text="start")],
                    )
                ],
            ),
        )
    )
    snapshot = store.load("session-1")
    assert snapshot is not None
    assert len(snapshot.state.permissions.grants) == 1

    async def reject_repeat(_decision: Ask) -> ApprovalResponse:
        raise AssertionError("exact Session permission should be reused")

    second_call = RuntimeMessage(
        role="assistant",
        content=[ToolUseBlock(id="call-2", name="mutate", input={"value": "same"})],
    )
    second = _runtime(Gateway([second_call, final]), store, reject_repeat)
    asyncio.run(
        _collect(
            second,
            ResumeQueryParams(
                session_id="session-1",
                workspace_root=str(tmp_path),
            ),
        )
    )

    assert approvals == 1
