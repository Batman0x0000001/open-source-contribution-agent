from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from osc_agent.runtime.context import ContextPipeline, GatewayContextSummarizer, MemoryToolResultStore, SessionTranscript
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.models import (
    QueryConfig,
    RuntimeMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
    WorktreeSession,
)


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
            working_directory=str(tmp_path),
        )
    )

    transcript_result = transcript.messages[-1].content[0]
    projection_result = projection.messages[-1].content[0]
    assert isinstance(transcript_result, ToolResultBlock)
    assert isinstance(projection_result, ToolResultBlock)
    assert transcript_result.content == original
    assert "Tool result persisted" in str(projection_result.content)
    assert store.read(session_id="session-1", result_id="call-1")


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
            working_directory="C:/repo",
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
            working_directory="C:/repo",
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
    context = ToolUseContext(
        session_id="session-1",
        working_directory=str(tmp_path / "worktree"),
        repository_root=str(tmp_path),
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
            working_directory=context.working_directory,
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
            working_directory="C:/repo",
        )
    )

    assert projection.summary_input_tokens == 7
    assert projection.summary_output_tokens == 3
