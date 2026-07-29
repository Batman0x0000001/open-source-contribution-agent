"""验证子 Agent 注册、上下文策略和共享 Runtime 不变量。"""

from __future__ import annotations

from osc_agent.runtime.state import CapabilityScope

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.runtime.messages import RuntimeMessage, TextBlock

from osc_agent.runtime.query import AgentRuntime
from osc_agent.runtime.session_store import FileSessionStore
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.subagents.models import SubagentDefinition, SubagentRequest
from osc_agent.subagents.registry import SubagentRegistration, SubagentRegistry
from osc_agent.subagents.runner import SubagentRunner


class EmptyArguments(ContractModel):
    pass


class TextResult(FrozenContractModel):
    value: str


class CompletingGateway:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        yield ModelCompleted(
            message=RuntimeMessage(role="assistant", content=[TextBlock(text="subagent result")]),
            stop_reason="end_turn",
        )


def definition(*, context_policy: str = "minimal") -> SubagentDefinition:
    return SubagentDefinition(
        name="explore",
        description="Explore the repository",
        system_prompt="Read and report.",
        capabilities=CapabilityScope(allowed_tools=frozenset({"agent", "read"})),
        context_policy=context_policy,
    )


def registration() -> SubagentRegistration:
    return SubagentRegistration(
        definition=definition(),
        input_model=EmptyArguments,
        output_model=TextResult,
        prompt_builder=lambda _arguments: "Return JSON.",
        read_only=True,
        concurrency_safe=True,
        max_parallel=2,
    )


def request(*, working_directory: str = "C:/repo", parent_messages=()) -> SubagentRequest:
    return SubagentRequest(
        prompt="inspect this",
        working_directory=working_directory,
        caller_capabilities=CapabilityScope(
            allowed_tools=frozenset({"agent", "read", "write"})
        ),
        parent_messages=tuple(parent_messages),
    )


def make_runner(
    gateway,
    *,
    ids: list[str] | None = None,
    session_store: FileSessionStore | None = None,
) -> SubagentRunner:
    registry = SubagentRegistry([registration()])
    generated_ids = iter(ids or [f"id-{index}" for index in range(20)])
    tools = ToolRegistry()
    runtime = AgentRuntime(
        QueryDependencies(
            model_gateway=gateway,
            tool_executor=ToolExecutor(tools),
            state_directory=(
                str(session_store.root.parent / "state")
                if session_store is not None
                else "C:/state"
            ),
            session_store=session_store,
        )
    )
    return SubagentRunner(
        runtime,
        registry,
        default_model="test-model",
        session_id_factory=lambda: next(generated_ids),
    )


def test_subagent_registry_is_sorted_strict_and_bounded() -> None:
    registry = SubagentRegistry([registration()])

    assert [item.definition.name for item in registry.list()] == ["explore"]
    assert registry.get("missing") is None
    with pytest.raises(ValueError, match="duplicate subagent"):
        registry.register(registration())
    with pytest.raises(TypeError, match="strict contract"):
        SubagentRegistration(
            definition=registration().definition,
            input_model=BaseModel,  # type: ignore[arg-type]
            output_model=TextResult,
            prompt_builder=lambda _arguments: "",
            read_only=True,
            concurrency_safe=True,
            max_parallel=1,
        )
    with pytest.raises(ValueError, match="explicitly enumerate"):
        SubagentRegistration(
            definition=definition().model_copy(
                update={"capabilities": CapabilityScope(allowed_tools=None)}
            ),
            input_model=EmptyArguments,
            output_model=TextResult,
            prompt_builder=lambda _arguments: "",
            read_only=True,
            concurrency_safe=True,
            max_parallel=1,
        )


def test_subagent_runner_reuses_runtime_and_removes_recursive_capability(tmp_path: Path) -> None:
    gateway = CompletingGateway()
    store = FileSessionStore(tmp_path / "sessions")
    runner = make_runner(gateway, ids=["child-session"], session_store=store)

    result = asyncio.run(
        runner.run("explore", request(working_directory=str(tmp_path)))
    )

    assert result.status == "completed"
    assert result.session_id == "child-session"
    assert result.output == "subagent result"
    assert gateway.requests[0].model == "test-model"
    snapshot = store.load("child-session")
    assert snapshot is not None
    assert snapshot.state.capabilities.allowed_tools == frozenset({"read"})


def test_fork_copies_parent_messages_and_minimal_ignores_them() -> None:
    parent = [RuntimeMessage(role="user", content=[TextBlock(text="parent context")])]

    fork_gateway = CompletingGateway()
    fork_runner = make_runner(fork_gateway)
    asyncio.run(
        fork_runner.run_definition(
            definition(context_policy="fork"),
            request(parent_messages=parent),
        )
    )
    assert fork_gateway.requests[0].messages[0].content[0].text == "parent context"
    assert fork_gateway.requests[0].messages[-1].content[0].text == "inspect this"
    assert len(parent) == 1

    minimal_gateway = CompletingGateway()
    minimal_runner = make_runner(minimal_gateway)
    asyncio.run(minimal_runner.run("explore", request(parent_messages=parent)))
    assert len(minimal_gateway.requests[0].messages) == 1
    assert minimal_gateway.requests[0].messages[0].content[0].text == "inspect this"


def test_fork_requires_parent_messages() -> None:
    runner = make_runner(CompletingGateway())

    with pytest.raises(ValueError, match="requires parent_messages"):
        asyncio.run(runner.run_definition(definition(context_policy="fork"), request()))


def test_dynamic_subagent_definition_cannot_retain_agent_tool(tmp_path: Path) -> None:
    gateway = CompletingGateway()
    store = FileSessionStore(tmp_path / "sessions")
    runner = make_runner(gateway, ids=["dynamic-child"], session_store=store)

    asyncio.run(
        runner.run_definition(
            definition(),
            request(working_directory=str(tmp_path)),
        )
    )

    snapshot = store.load("dynamic-child")
    assert snapshot is not None
    assert snapshot.state.capabilities.allowed_tools == frozenset({"read"})
