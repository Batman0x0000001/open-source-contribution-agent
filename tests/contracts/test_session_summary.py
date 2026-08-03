"""验证 Session V6 摘要只使用成功的类型化 Tool 证据。"""

from osc_agent.application.session_summary import build_session_summary
from osc_agent.runtime.messages import RuntimeMessage, ToolResultBlock, ToolUseBlock
from osc_agent.runtime.session import SessionMetadata, SessionSnapshot
from tests.runtime_factories import agent_run_state


def snapshot(messages=None) -> SessionSnapshot:
    return SessionSnapshot(
        metadata=SessionMetadata(
            schema_version=6,
            session_id="session-1",
            workspace_root="C:/repo",
            model="test",
        ),
        state=agent_run_state("C:/repo", status="completed"),
        messages=messages or [],
    )


def test_summary_uses_only_successful_typed_tool_evidence() -> None:
    value = snapshot([
        RuntimeMessage(role="assistant", content=[
            ToolUseBlock(id="write", name="write_file", input={"path": "src/app.py", "content": "secret"}),
            ToolUseBlock(id="test", name="bash", input={"command": "python -m pytest"}),
            ToolUseBlock(id="snapshot", name="git_diff", input={}),
        ]),
        RuntimeMessage(role="user", content=[
            ToolResultBlock(tool_use_id="write", content={"data": {"path": "src/app.py"}, "error": None}),
            ToolResultBlock(tool_use_id="test", content={"data": {"command": "python -m pytest", "command_kind": "test", "success": True}, "error": None}),
            ToolResultBlock(tool_use_id="snapshot", content={"data": {"files": [{"path": "src/app.py"}], "truncated": False, "has_changes": True}, "error": None}),
        ]),
    ])
    summary = build_session_summary(value)
    assert summary.touched_files == ["src/app.py"]
    assert summary.last_test_success is True
    assert summary.snapshot_files == ["src/app.py"]
    assert summary.snapshot_complete is True


def test_summary_does_not_infer_evidence_from_assistant_text() -> None:
    summary = build_session_summary(snapshot())
    assert summary.last_test_success is None
    assert summary.verification_verdict is None
    assert summary.snapshot_complete is None
