"""验证Runtime 上下文管线的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from osc_agent.runtime.context import (
    ContextPipeline,
    ContextSummary,
    GatewayContextSummarizer,
    SessionTranscript,
)
from osc_agent.runtime.session_store import FileToolResultStore, MemoryToolResultStore
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.messages import RuntimeMessage, TextBlock, ToolResultBlock, ToolUseBlock
from osc_agent.runtime.query_models import QueryConfig
from osc_agent.workspaces.models import WorktreeSession


def message(role: str, *blocks):
    return RuntimeMessage(role=role, content=list(blocks))


def test_projection_compacts_without_mutating_authoritative_transcript(tmp_path: Path) -> None:
    original = "x" * 1_000
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[
            message("user", TextBlock(text="start")),
            message("assistant", ToolUseBlock(id="call-1", name="read", input={})),
            message("user", ToolResultBlock(tool_use_id="call-1", content=original)),
        ],
    )
    store = MemoryToolResultStore()
    pipeline = ContextPipeline(tool_result_store=store)

    projection = asyncio.run(
        pipeline.project(
            transcript,
            config=QueryConfig(max_tool_result_chars=100),
        )
    )

    transcript_result = transcript.messages[-1].content[0]
    projection_result = projection.messages[-1].content[0]
    assert isinstance(transcript_result, ToolResultBlock)
    assert isinstance(projection_result, ToolResultBlock)
    assert transcript_result.content == original
    assert "Tool result persisted" in str(projection_result.content)
    assert store.read(session_id="session-1", result_id="call-1")


def test_repeated_projection_reuses_the_same_persisted_tool_result(tmp_path: Path) -> None:
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[
            message("assistant", ToolUseBlock(id="call-1", name="read", input={})),
            message("user", ToolResultBlock(tool_use_id="call-1", content="x" * 1_000)),
        ],
    )
    result_root = tmp_path / "tool-results"
    pipeline = ContextPipeline(tool_result_store=FileToolResultStore(result_root))

    first = asyncio.run(
        pipeline.project(transcript, config=QueryConfig(max_tool_result_chars=100))
    )
    second = asyncio.run(
        pipeline.project(transcript, config=QueryConfig(max_tool_result_chars=100))
    )

    assert first.messages[-1].content[0].content == second.messages[-1].content[0].content
    assert [path.name for path in (result_root / "session-1").iterdir()] == [
        "call-1.txt"
    ]


def test_read_file_result_stays_inline_instead_of_entering_a_readback_loop() -> None:
    original = "README contents\n" * 1_000
    store = MemoryToolResultStore()
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[
            message(
                "assistant",
                ToolUseBlock(
                    id="read-call",
                    name="read_file",
                    input={"path": "README.md"},
                ),
            ),
            message(
                "user",
                ToolResultBlock(tool_use_id="read-call", content=original),
            ),
        ],
    )

    projection = asyncio.run(
        ContextPipeline(tool_result_store=store).project(
            transcript,
            config=QueryConfig(max_tool_result_chars=100),
        )
    )

    projected = projection.messages[-1].content[0]
    assert isinstance(projected, ToolResultBlock)
    assert projected.content == original
    try:
        store.read(session_id="session-1", result_id="read-call")
    except ValueError:
        pass
    else:
        raise AssertionError("read_file output must not be persisted for model readback")


def test_micro_compact_keeps_bounded_read_file_results_inline() -> None:
    messages = []
    for index in range(4):
        call_id = f"read-{index}"
        messages.extend(
            [
                message(
                    "assistant",
                    ToolUseBlock(
                        id=call_id,
                        name="read_file",
                        input={"path": f"file-{index}.txt"},
                    ),
                ),
                message(
                    "user",
                    ToolResultBlock(
                        tool_use_id=call_id,
                        content=f"original-{index}-" + "x" * 500,
                    ),
                ),
            ]
        )
    transcript = SessionTranscript(session_id="session-1", messages=messages)

    projection = asyncio.run(
        ContextPipeline().project(
            transcript,
            config=QueryConfig(
                max_tool_result_chars=100_000,
                keep_recent_tool_results=1,
                auto_compact_chars=100_000,
            ),
        )
    )

    first_result = projection.messages[1].content[0]
    assert isinstance(first_result, ToolResultBlock)
    assert str(first_result.content).startswith("original-0-")


def test_projection_does_not_persist_a_retrieved_tool_result_again() -> None:
    store = MemoryToolResultStore()
    original_id = store.persist(
        session_id="session-1",
        tool_use_id="original-call",
        content="original result",
    )
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[
            message(
                "assistant",
                ToolUseBlock(
                    id="read-call",
                    name="read_tool_result",
                    input={"result_id": original_id},
                ),
            ),
            message(
                "user",
                ToolResultBlock(tool_use_id="read-call", content="x" * 1_000),
            ),
        ],
    )

    projection = asyncio.run(
        ContextPipeline(tool_result_store=store).project(
            transcript,
            config=QueryConfig(max_tool_result_chars=100),
        )
    )

    projected = projection.messages[-1].content[0]
    assert isinstance(projected, ToolResultBlock)
    assert "Retrieved tool result page compacted" in str(projected.content)
    assert "Tool result persisted" not in str(projected.content)
    try:
        store.read(session_id="session-1", result_id="read-call")
    except ValueError:
        pass
    else:
        raise AssertionError("retrieved tool result was recursively persisted")


def test_auto_compact_preserves_tool_use_result_pair() -> None:
    tool_use = ToolUseBlock(id="call-1", name="read", input={})
    tool_result = ToolResultBlock(tool_use_id="call-1", content="result")
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[
            message("user", TextBlock(text="old context" * 20)),
            message("assistant", tool_use),
            message("user", tool_result),
        ],
    )

    projection = asyncio.run(
        ContextPipeline().project(
            transcript,
            config=QueryConfig(auto_compact_chars=1),
        )
    )

    projected_uses = {
        block.id
        for item in projection.messages
        for block in item.content
        if isinstance(block, ToolUseBlock)
    }
    projected_results = {
        block.tool_use_id
        for item in projection.messages
        for block in item.content
        if isinstance(block, ToolResultBlock)
    }
    assert projection.compacted is True
    assert projected_uses == projected_results == {"call-1"}


def test_reactive_compact_can_replace_the_entire_projection() -> None:
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[message("user", TextBlock(text="large prompt"))],
    )

    projection = asyncio.run(
        ContextPipeline().project(
            transcript,
            config=QueryConfig(),
            force_reason="reactive_compact",
        )
    )

    assert len(projection.messages) == 1
    assert projection.reason == "reactive_compact"
    assert "Context compacted: reactive_compact" in projection.messages[0].content[0].text


def test_runtime_reminder_reinjects_plan_and_worktree_without_mutating_transcript(tmp_path: Path) -> None:
    plans = tmp_path / "state" / "plans"
    plans.mkdir(parents=True)
    (plans / "session-1.md").write_text("# Approved plan", encoding="utf-8")
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[message("user", TextBlock(text="continue"))],
    )
    context = tool_context(
        session_id="session-1",
        working_directory=str(tmp_path / "worktree"),
        state_directory=str(tmp_path / "state"),
        permission_mode="default",
        plan_path="session-1.md",
        worktree=WorktreeSession(
            path=str(tmp_path / "worktree"),
            original_working_directory=str(tmp_path),
            branch="osc-agent/test",
            base_commit="abc123",
        ),
    )

    projection = asyncio.run(
        ContextPipeline().project(
            transcript,
            config=QueryConfig(),
            runtime_context=context,
        )
    )

    assert "# Approved plan" in projection.system_reminder
    assert "osc-agent/test" in projection.system_reminder
    assert transcript.messages[0].content[0].text == "continue"


def test_model_summary_reports_its_token_usage() -> None:
    class SummaryGateway:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            yield ModelCompleted(
                message=RuntimeMessage(role="assistant", content=[TextBlock(text="Preserved objective and next action.")]),
                stop_reason="end_turn",
                input_tokens=7,
                output_tokens=3,
            )

    transcript = SessionTranscript(
        session_id="session-1",
        messages=[
            message("user", TextBlock(text="old objective")),
            message("assistant", TextBlock(text="old response")),
            message("user", TextBlock(text="new objective")),
        ],
    )
    projection = asyncio.run(
        ContextPipeline(
            summarizer=GatewayContextSummarizer(SummaryGateway(), model="test")
        ).project(
            transcript,
            config=QueryConfig(auto_compact_chars=1),
        )
    )

    assert projection.summary_input_tokens == 7
    assert projection.summary_output_tokens == 3


def test_auto_compact_summarizes_authoritative_tool_results_before_projection_trimming() -> None:
    class RecordingSummarizer:
        def __init__(self) -> None:
            self.messages: list[RuntimeMessage] = []

        async def summarize(
            self,
            messages: list[RuntimeMessage],
            *,
            reason: str,
        ) -> ContextSummary:
            self.messages = messages
            return ContextSummary(text=f"summary for {reason}")

    original = "authoritative README content\n" * 100
    summarizer = RecordingSummarizer()
    transcript = SessionTranscript(
        session_id="session-1",
        messages=[
            message("user", TextBlock(text="translate README")),
            message(
                "assistant",
                ToolUseBlock(id="read-1", name="read", input={}),
            ),
            message("user", ToolResultBlock(tool_use_id="read-1", content=original)),
            message("assistant", TextBlock(text="middle one")),
            message("user", TextBlock(text="continue one")),
            message("assistant", TextBlock(text="middle two")),
            message("user", TextBlock(text="continue two")),
            message("assistant", TextBlock(text="latest")),
            message("user", TextBlock(text="continue latest")),
        ],
    )

    asyncio.run(
        ContextPipeline(summarizer=summarizer).project(
            transcript,
            config=QueryConfig(
                max_tool_result_chars=100,
                keep_recent_tool_results=1,
                auto_compact_chars=1,
            ),
        )
    )

    summarized_results = [
        block.content
        for item in summarizer.messages
        for block in item.content
        if isinstance(block, ToolResultBlock)
    ]
    assert original in summarized_results
