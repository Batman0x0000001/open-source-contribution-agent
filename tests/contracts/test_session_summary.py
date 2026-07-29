"""验证会话摘要的契约、边界条件与回归行为。"""

from __future__ import annotations

from osc_agent.application.session_summary import build_session_summary
from osc_agent.runtime.messages import RuntimeMessage, ToolResultBlock, ToolUseBlock
from osc_agent.runtime.session import SessionMetadata, SessionRuntimeState, SessionSnapshot


def test_summary_uses_only_successful_typed_tool_evidence() -> None:
    snapshot = SessionSnapshot(
        metadata=SessionMetadata(
            schema_version=4,
            session_id="session-1",
            repository_root="C:/repo",
            initial_working_directory="C:/repo",
            model="test",
        ),
        runtime_state=SessionRuntimeState(
            last_status="completed",
            last_reason="end_turn",
        ),
        messages=[
            RuntimeMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="write",
                        name="write_file",
                        input={"path": "src/app.py", "content": "secret"},
                    ),
                    ToolUseBlock(
                        id="test",
                        name="bash",
                        input={"command": "python -m pytest"},
                    ),
                    ToolUseBlock(id="snapshot", name="git_diff", input={}),
                ],
            ),
            RuntimeMessage(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="write",
                        content={"data": {"path": "src/app.py"}, "error": None},
                    ),
                    ToolResultBlock(
                        tool_use_id="test",
                        content={
                            "data": {
                                "command": "python -m pytest",
                                "command_kind": "test",
                                "success": True,
                            },
                            "error": None,
                        },
                    ),
                    ToolResultBlock(
                        tool_use_id="snapshot",
                        content={
                            "data": {
                                "files": [{"path": "src/app.py"}],
                                "truncated": False,
                                "has_changes": True,
                            },
                            "error": None,
                        },
                    ),
                ],
            ),
        ],
    )

    summary = build_session_summary(snapshot)

    assert summary.touched_files == ["src/app.py"]
    assert summary.last_test_success is True
    assert summary.snapshot_files == ["src/app.py"]
    assert summary.snapshot_complete is True


def test_summary_does_not_infer_evidence_from_assistant_text() -> None:
    snapshot = SessionSnapshot(
        metadata=SessionMetadata(
            schema_version=4,
            session_id="session-1",
            repository_root="C:/repo",
            initial_working_directory="C:/repo",
            model="test",
        )
    )

    summary = build_session_summary(snapshot)

    assert summary.last_test_success is None
    assert summary.verification_verdict is None
    assert summary.snapshot_complete is None
