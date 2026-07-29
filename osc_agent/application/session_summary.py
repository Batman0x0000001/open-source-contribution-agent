"""生成会话压缩摘要并保留关键运行证据。"""

from __future__ import annotations

from pydantic import Field

from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.messages import ToolResultBlock, ToolUseBlock
from osc_agent.runtime.session import SessionSnapshot


class SessionSummary(FrozenContractModel):
    session_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    reason: str | None = None
    working_directory: str = Field(min_length=1)
    worktree_path: str | None = None
    worktree_branch: str | None = None
    touched_files: list[str] = Field(default_factory=list)
    last_test_command: str | None = None
    last_test_success: bool | None = None
    verification_verdict: str | None = None
    snapshot_files: list[str] = Field(default_factory=list)
    snapshot_complete: bool | None = None
    snapshot_has_changes: bool | None = None


def build_session_summary(snapshot: SessionSnapshot) -> SessionSummary:
    calls: dict[str, ToolUseBlock] = {}
    touched_files: list[str] = []
    last_test_command: str | None = None
    last_test_success: bool | None = None
    verification_verdict: str | None = None
    snapshot_files: list[str] = []
    snapshot_complete: bool | None = None
    snapshot_has_changes: bool | None = None

    for message in snapshot.messages:
        if message.role == "assistant":
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    calls[block.id] = block
            continue
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            call = calls.get(block.tool_use_id)
            content = block.content if isinstance(block.content, dict) else {}
            if call is None or block.is_error or content.get("error") is not None:
                continue
            data = content.get("data")
            if not isinstance(data, dict):
                continue
            if call.name in {"write_file", "edit_file"}:
                path = call.input.get("path")
                if isinstance(path, str) and path not in touched_files:
                    touched_files.append(path)
            elif call.name == "bash" and data.get("command_kind") == "test":
                command = data.get("command")
                success = data.get("success")
                if isinstance(command, str) and isinstance(success, bool):
                    last_test_command = command
                    last_test_success = success
            elif call.name == "agent":
                result = data.get("result")
                if (
                    isinstance(result, dict)
                    and result.get("evidence_type") == "independent_verification"
                    and isinstance(result.get("verdict"), str)
                ):
                    verification_verdict = result["verdict"]
            elif call.name == "git_diff":
                files = data.get("files")
                if isinstance(files, list):
                    snapshot_files = [
                        str(item["path"])
                        for item in files
                        if isinstance(item, dict) and isinstance(item.get("path"), str)
                    ]
                truncated = data.get("truncated")
                if isinstance(truncated, bool):
                    snapshot_complete = not truncated
                has_changes = data.get("has_changes")
                if isinstance(has_changes, bool):
                    snapshot_has_changes = has_changes

    run_state = snapshot.state
    worktree = run_state.workspace.worktree
    return SessionSummary(
        session_id=snapshot.metadata.session_id,
        status=run_state.last_status or "unknown",
        reason=run_state.last_reason,
        working_directory=run_state.workspace.working_directory,
        worktree_path=worktree.path if worktree is not None else None,
        worktree_branch=worktree.branch if worktree is not None else None,
        touched_files=touched_files,
        last_test_command=last_test_command,
        last_test_success=last_test_success,
        verification_verdict=verification_verdict,
        snapshot_files=snapshot_files,
        snapshot_complete=snapshot_complete,
        snapshot_has_changes=snapshot_has_changes,
    )
