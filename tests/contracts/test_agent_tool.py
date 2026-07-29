"""验证 agent Tool 的协议、限流、取消和工作区保护。"""

from __future__ import annotations

from osc_agent.runtime.state import CapabilityScope

from tests.runtime_factories import tool_context

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from osc_agent.runtime.dependencies import QueryDependencies
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.runtime.events import RunStopped
from osc_agent.runtime.messages import RuntimeMessage, TextBlock, ToolResultBlock, ToolUseBlock
from osc_agent.runtime.query_models import StartQueryParams

from osc_agent.runtime.query import AgentRuntime
from osc_agent.configuration.runtime import default_runtime_config_path, load_runtime_config
from osc_agent.runtime.session_store import FileSessionStore
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.subagents.builtins.explore import build_explore_subagent
from osc_agent.subagents.models import SubagentDefinition, SubagentRequest, SubagentRunResult
from osc_agent.subagents.registry import SubagentRegistration, SubagentRegistry
from osc_agent.subagents.runner import SubagentRunner
from osc_agent.subagents.tool import AgentTool, AgentToolInput


EXPLORE_CONFIG = load_runtime_config(default_runtime_config_path()).agents.explore.to_query_config()


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=root, check=True)


class EmptyArguments(ContractModel):
    pass


class TextResult(FrozenContractModel):
    value: str


def registration(*, max_parallel: int = 2) -> SubagentRegistration:
    return SubagentRegistration(
        definition=SubagentDefinition(
            name="explore",
            description="Explore the repository",
            system_prompt="Read and report.",
            capabilities=CapabilityScope(allowed_tools=frozenset({"read"})),
        ),
        input_model=EmptyArguments,
        output_model=TextResult,
        prompt_builder=lambda _arguments: "Return JSON.",
        read_only=True,
        concurrency_safe=True,
        max_parallel=max_parallel,
    )


def context(repo: Path) -> tool_context:
    return tool_context(
        session_id="parent-1",
        working_directory=str(repo),
        state_directory=str(repo / ".state"),
        capabilities=CapabilityScope(allowed_tools=frozenset({"agent", "read_file"})),
        transcript_messages=[
            RuntimeMessage(role="user", content=[TextBlock(text="private parent history")])
        ],
    )


def test_agent_tool_validates_explore_input_and_typed_output(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    source = tmp_path / "src.py"
    source.write_text("value = 1\n", encoding="utf-8")

    class RecordingRunner:
        def __init__(self) -> None:
            self.name: str | None = None
            self.request: SubagentRequest | None = None

        async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
            self.name = name
            self.request = request
            return SubagentRunResult(
                session_id="child-1",
                status="completed",
                output=(
                    '{"summary":"found","findings":[{"claim":"value exists",'
                    '"confidence":"high","evidence":[{"path":"src.py","line_start":1,'
                    '"observation":"assignment"}]}],"relevant_files":["src.py"],'
                    '"likely_change_locations":[],"recommended_tests":[],"unresolved_questions":[]}'
                ),
            )

    registry = SubagentRegistry(
        [build_explore_subagent(model="test-model", config=EXPLORE_CONFIG)]
    )
    recorder = RecordingRunner()
    tool = AgentTool(recorder, registry)  # type: ignore[arg-type]

    result = asyncio.run(
        tool.call(
            AgentToolInput(
                agent="explore",
                task="Find the assignment",
                arguments={"scope_paths": ["."], "context_summary": "Small fixture"},
            ),
            context(tmp_path),
        )
    )

    assert result.data["status"] == "completed"
    assert result.data["result"]["findings"][0]["evidence"][0]["path"] == "src.py"
    assert recorder.name == "explore"
    assert recorder.request is not None
    assert "private parent history" not in recorder.request.prompt


def test_agent_tool_returns_specific_errors_for_unknown_input_and_output(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    class InvalidRunner:
        async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
            return SubagentRunResult(
                session_id="child-1",
                status="completed",
                output='{"summary":"unsupported","findings":[],"relevant_files":[],"likely_change_locations":[],"recommended_tests":[],"unresolved_questions":[]}',
            )

    registry = SubagentRegistry(
        [build_explore_subagent(model="test-model", config=EXPLORE_CONFIG)]
    )
    tool = AgentTool(InvalidRunner(), registry)  # type: ignore[arg-type]

    unknown = asyncio.run(
        tool.call(AgentToolInput(agent="missing", task="inspect"), context(tmp_path))
    )
    invalid_input = asyncio.run(
        tool.call(
            AgentToolInput(
                agent="explore",
                task="inspect",
                arguments={"scope_paths": ["../outside"]},
            ),
            context(tmp_path),
        )
    )
    invalid_output = asyncio.run(
        tool.call(AgentToolInput(agent="explore", task="inspect"), context(tmp_path))
    )

    assert unknown.error and unknown.error.code == "AGENT_NOT_FOUND"
    assert invalid_input.error and invalid_input.error.code == "AGENT_INPUT_INVALID"
    assert invalid_output.error and invalid_output.error.code == "AGENT_OUTPUT_INVALID"

    source = tmp_path / "one-line.py"
    source.write_text("value = 1\n", encoding="utf-8")
    output_model = registry.get("explore").output_model
    with pytest.raises(ValidationError, match="exceeds one-line.py line count"):
        output_model.model_validate(
            {
                "summary": "bad line",
                "findings": [
                    {
                        "claim": "outside file",
                        "confidence": "low",
                        "evidence": [
                            {
                                "path": "one-line.py",
                                "line_start": 2,
                                "observation": "not present",
                            }
                        ],
                    }
                ],
                "relevant_files": ["one-line.py"],
                "likely_change_locations": [],
                "recommended_tests": [],
                "unresolved_questions": [],
            },
            context={"repository_root": tmp_path},
        )


def test_agent_tool_limits_parallel_runs_and_propagates_cancellation(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    class ControlledRunner:
        def __init__(self) -> None:
            self.active = 0
            self.maximum = 0
            self.two_started = asyncio.Event()
            self.release = asyncio.Event()

        async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            if self.active == 2:
                self.two_started.set()
            try:
                await self.release.wait()
                return SubagentRunResult(
                    session_id=request.prompt[-8:] or "child",
                    status="completed",
                    output='{"value":"done"}',
                )
            finally:
                self.active -= 1

    async def exercise_parallelism() -> int:
        runner = ControlledRunner()
        registry = SubagentRegistry([registration(max_parallel=2)])
        tool = AgentTool(runner, registry)  # type: ignore[arg-type]
        calls = [
            asyncio.create_task(
                tool.call(
                    AgentToolInput(agent="explore", task=f"inspect-{index}"),
                    context(tmp_path),
                )
            )
            for index in range(3)
        ]
        await asyncio.wait_for(runner.two_started.wait(), timeout=5)
        assert runner.active == 2
        runner.release.set()
        await asyncio.gather(*calls)
        return runner.maximum

    assert asyncio.run(exercise_parallelism()) == 2

    class CancellingRunner:
        async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def exercise_cancellation() -> None:
        tool = AgentTool(
            CancellingRunner(),  # type: ignore[arg-type]
            SubagentRegistry([registration()]),
        )
        pending = asyncio.create_task(
            tool.call(AgentToolInput(agent="explore", task="wait"), context(tmp_path))
        )
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(exercise_cancellation())


def test_one_parallel_agent_failure_does_not_cancel_its_sibling(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    class MixedRunner:
        async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
            await asyncio.sleep(0)
            if "fail" in request.prompt:
                return SubagentRunResult(
                    session_id="failed-child",
                    status="failed",
                    error="bounded exploration failed",
                )
            return SubagentRunResult(
                session_id="successful-child",
                status="completed",
                output='{"value":"evidence"}',
            )

    async def scenario():
        tool = AgentTool(
            MixedRunner(),  # type: ignore[arg-type]
            SubagentRegistry([registration()]),
        )
        return await asyncio.gather(
            tool.call(AgentToolInput(agent="explore", task="fail"), context(tmp_path)),
            tool.call(AgentToolInput(agent="explore", task="succeed"), context(tmp_path)),
        )

    failed, succeeded = asyncio.run(scenario())

    assert failed.data["status"] == "failed"
    assert failed.data["child_session_id"] == "failed-child"
    assert succeeded.data["status"] == "completed"
    assert succeeded.data["result"] == {"value": "evidence"}


def test_parent_cancellation_stops_explore_and_persists_tool_pair(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    child_started = asyncio.Event()

    class BlockingGateway:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            if request.system_prompt.startswith("parent"):
                yield ModelCompleted(
                    message=RuntimeMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="agent-1",
                                name="agent",
                                input={
                                    "agent": "explore",
                                    "task": "wait for cancellation",
                                    "arguments": {},
                                },
                            )
                        ],
                    ),
                    stop_reason="tool_use",
                )
                return
            child_started.set()
            await asyncio.Event().wait()

    store = FileSessionStore(tmp_path.parent / f"{tmp_path.name}-sessions")
    tools = ToolRegistry()
    executor = ToolExecutor(tools)
    runtime = AgentRuntime(
        QueryDependencies(
            model_gateway=BlockingGateway(),
            tool_executor=executor,
            state_directory=str(tmp_path / "state"),
            session_store=store,
        )
    )
    registry = SubagentRegistry([registration()])
    runner = SubagentRunner(runtime, registry, default_model="test-model")
    tools.register(AgentTool(runner, registry))

    async def scenario():
        async def collect():
            return [
                event
                async for event in runtime.query(
                    StartQueryParams(
                        session_id="parent-session",
                        model="test-model",
                        system_prompt="parent",
                        messages=[
                            RuntimeMessage(
                                role="user",
                                content=[TextBlock(text="start")],
                            )
                        ],
                        workspace_root=str(tmp_path),
                        capabilities=CapabilityScope(
                            allowed_tools=frozenset({"agent", "read"})
                        ),
                    )
                )
            ]

        pending = asyncio.create_task(collect())
        await asyncio.wait_for(child_started.wait(), timeout=5)
        pending.cancel()
        return await pending

    events = asyncio.run(scenario())
    snapshot = store.load("parent-session")

    assert isinstance(events[-1], RunStopped)
    assert events[-1].transition.kind == "cancelled"
    assert snapshot is not None
    assert any(
        isinstance(block, ToolResultBlock) and block.tool_use_id == "agent-1"
        for message in snapshot.messages
        for block in message.content
    )
