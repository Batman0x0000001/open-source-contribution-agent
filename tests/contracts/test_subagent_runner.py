"""验证子 Agent 注册、上下文策略和共享 Runtime 不变量。"""

from __future__ import annotations

from osc_agent.runtime.state import CapabilityScope

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import subprocess

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
from osc_agent.subagents.models import (
    SubagentDefinition,
    SubagentRequest,
    SubagentRunResult,
)
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


def definition(*, allowed_tools: frozenset[str] = frozenset({"read"})) -> SubagentDefinition:
    return SubagentDefinition(
        name="explore",
        description="Explore the repository",
        system_prompt="Read and report.",
        capabilities=CapabilityScope(allowed_tools=allowed_tools),
    )


def registration(
    *,
    max_parallel: int = 2,
    read_only: bool = True,
) -> SubagentRegistration:
    return SubagentRegistration(
        definition=definition(),
        input_model=EmptyArguments,
        output_model=TextResult,
        prompt_builder=lambda _arguments: "Return JSON.",
        read_only=read_only,
        concurrency_safe=True,
        max_parallel=max_parallel,
    )


def request(*, working_directory: str = "C:/repo") -> SubagentRequest:
    return SubagentRequest(
        prompt="inspect this",
        working_directory=working_directory,
        caller_capabilities=CapabilityScope(
            allowed_tools=frozenset({"agent", "read", "write"})
        ),
    )


def make_runner(
    gateway,
    *,
    ids: list[str] | None = None,
    session_store: FileSessionStore | None = None,
    registered: SubagentRegistration | None = None,
) -> SubagentRunner:
    registry = SubagentRegistry([registered or registration()])
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


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--quiet", "--allow-empty", "-m", "initial"],
        cwd=root,
        check=True,
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
    with pytest.raises(ValueError, match="recursive agent capability"):
        SubagentRegistration(
            definition=definition(allowed_tools=frozenset({"agent", "read"})),
            input_model=EmptyArguments,
            output_model=TextResult,
            prompt_builder=lambda _arguments: "",
            read_only=True,
            concurrency_safe=True,
            max_parallel=1,
        )


def test_subagent_registry_rejects_unknown_declared_tools() -> None:
    registry = SubagentRegistry(
        [
            SubagentRegistration(
                definition=definition(allowed_tools=frozenset({"read", "missing"})),
                input_model=EmptyArguments,
                output_model=TextResult,
                prompt_builder=lambda _arguments: "",
                read_only=True,
                concurrency_safe=True,
                max_parallel=1,
            )
        ]
    )

    with pytest.raises(ValueError, match="unknown tools: missing"):
        registry.validate_allowed_tools({"read"})


def test_subagent_runner_reuses_runtime_and_narrows_capability(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    gateway = CompletingGateway()
    store = FileSessionStore(tmp_path.parent / f"{tmp_path.name}-sessions")
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


def test_subagent_uses_only_its_bounded_task_prompt(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    gateway = CompletingGateway()
    runner = make_runner(gateway)

    asyncio.run(runner.run("explore", request(working_directory=str(tmp_path))))

    assert len(gateway.requests[0].messages) == 1
    assert gateway.requests[0].messages[0].content[0].text == "inspect this"
    assert not hasattr(runner, "run_definition")


def test_runner_owns_registration_parallelism(monkeypatch) -> None:
    runner = make_runner(
        CompletingGateway(),
        registered=registration(max_parallel=2, read_only=False),
    )
    active = 0
    maximum = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_execute(session_id, definition, subagent_request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            two_started.set()
        try:
            await release.wait()
            return SubagentRunResult(
                session_id=session_id,
                status="completed",
                output="done",
            )
        finally:
            active -= 1

    monkeypatch.setattr(runner, "_execute", controlled_execute)

    async def exercise() -> None:
        calls = [
            asyncio.create_task(runner.run("explore", request()))
            for _index in range(3)
        ]
        await asyncio.wait_for(two_started.wait(), timeout=5)
        assert active == 2
        release.set()
        await asyncio.gather(*calls)

    asyncio.run(exercise())
    assert maximum == 2


def test_cancellation_remains_authoritative_when_read_only_guard_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    runner = make_runner(CompletingGateway())
    started = asyncio.Event()

    async def mutating_execute(session_id, definition, subagent_request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            (tmp_path / "unexpected.txt").write_text("changed", encoding="utf-8")

    monkeypatch.setattr(runner, "_execute", mutating_execute)

    async def exercise() -> None:
        pending = asyncio.create_task(
            runner.run("explore", request(working_directory=str(tmp_path)))
        )
        await started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(exercise())
    assert (tmp_path / "unexpected.txt").is_file()
