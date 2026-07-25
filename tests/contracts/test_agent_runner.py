from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from osc_agent.agents.definitions import AgentDefinition, AgentInvocation, AgentRunResult
from osc_agent.agents.runner import AgentRunner
from osc_agent.agents.tool import AgentTool, AgentToolInput
from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.models import CapabilityScope, RuntimeMessage, TextBlock, ToolUseContext
from osc_agent.runtime.query import AgentRuntime
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor


class CompletingGateway:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelCompleted(
            message=RuntimeMessage(role="assistant", content=[TextBlock(text="agent result")]),
            stop_reason="end_turn",
        )


def make_runner(gateway, *, ids: list[str] | None = None) -> AgentRunner:
    registry = ToolRegistry()
    generated_ids = iter(ids or [f"id-{index}" for index in range(20)])
    runtime = AgentRuntime(
        QueryDependencies(
            model_gateway=gateway,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            new_id=lambda: next(generated_ids),
        )
    )
    definition = AgentDefinition(
        name="explore",
        description="Explore the repository",
        system_prompt="Read and report.",
        capabilities=CapabilityScope(allowed_tools=frozenset({"read"})),
    )
    return AgentRunner(runtime, [definition], default_model="test-model")


def invocation(*, mode: str = "inline", parent_messages=None) -> AgentInvocation:
    return AgentInvocation(
        agent_name="explore",
        prompt="inspect this",
        mode=mode,
        parent_session_id="parent-1",
        working_directory="C:/repo",
        caller_capabilities=CapabilityScope(allowed_tools=frozenset({"read", "write"})),
        parent_messages=parent_messages or [],
    )


def test_agent_runner_reuses_runtime_with_isolated_session() -> None:
    gateway = CompletingGateway()
    runner = make_runner(gateway, ids=["task-1", "child-session"])

    result = asyncio.run(runner.run(invocation()))

    assert result.status == "completed"
    assert result.session_id == "child-session"
    assert result.output == "agent result"
    assert gateway.requests[0].model == "test-model"


def test_fork_inherits_a_copy_of_parent_messages() -> None:
    gateway = CompletingGateway()
    runner = make_runner(gateway)
    parent = [RuntimeMessage(role="user", content=[TextBlock(text="parent context")])]

    asyncio.run(runner.run(invocation(mode="fork", parent_messages=parent)))

    assert gateway.requests[0].messages[0].content[0].text == "parent context"
    assert gateway.requests[0].messages[-1].content[0].text == "inspect this"
    assert len(parent) == 1


def test_agent_tool_delegates_fork_execution_to_runner() -> None:
    class RecordingRunner:
        def __init__(self) -> None:
            self.invocation: AgentInvocation | None = None

        async def run(self, invocation: AgentInvocation) -> AgentRunResult:
            self.invocation = invocation
            return AgentRunResult(
                task_id="task-1",
                session_id="child-1",
                status="completed",
                output="done",
            )

    recorder = RecordingRunner()
    tool = AgentTool(recorder, WorktreeManager(Path("C:/state/worktrees")))  # type: ignore[arg-type]
    parent_message = RuntimeMessage(role="user", content=[TextBlock(text="parent")])
    context = ToolUseContext(
        session_id="parent-1",
        working_directory="C:/repo",
        repository_root="C:/repo",
        state_directory="C:/state",
        capabilities=CapabilityScope(allowed_tools=frozenset({"read"})),
        transcript_messages=[parent_message],
    )

    result = asyncio.run(
        tool.call(
            AgentToolInput(agent="explore", prompt="inspect", mode="fork"),
            context,
        )
    )

    assert result.data["status"] == "completed"
    assert recorder.invocation is not None
    assert recorder.invocation.parent_messages == [parent_message]
    assert recorder.invocation.caller_capabilities == context.capabilities
