"""验证文件系统工具的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import apply_tool_result, tool_context

import asyncio
from pathlib import Path

import pytest

from osc_agent.runtime.messages import RuntimeMessage, ToolResultBlock, ToolUseBlock
from osc_agent.runtime.tool_models import ApprovalResponse, Ask
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor
from tests.contracts.registry_factory import build_test_tool_registry
from osc_agent.tools.filesystem import (
    EditFileInput,
    GlobInput,
    ReadFileInput,
    ReadFileTool,
    WriteFileInput,
)


def context(root: Path) -> tool_context:
    return tool_context(session_id="session-1", working_directory=str(root), state_directory=str(root / "state"))


def test_core_registry_has_one_authoritative_definition_per_migrated_tool() -> None:
    registry = build_test_tool_registry()

    assert registry.get("read_file").input_model is ReadFileInput
    assert registry.get("write_file").input_model is WriteFileInput
    assert registry.get("edit_file").input_model is EditFileInput
    assert registry.get("glob").input_model is GlobInput
    schemas = registry.schemas(context(Path("C:/repo")))
    assert all(schema["description"].strip() for schema in schemas)
    assert [schema["name"] for schema in schemas] == [
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "bash",
        "git_status",
        "git_diff",
        "git_log",
        "github_list_issues",
        "github_get_issue",
        "ask_user_question",
        "enter_plan_mode",
        "write_plan",
        "read_plan",
        "exit_plan_mode",
        "enter_worktree",
        "exit_worktree",
        "read_tool_result",
    ]


def test_read_file_is_input_sensitive_behavior_object(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("abcdef", encoding="utf-8")
    tool = ReadFileTool()
    executor = ToolExecutor(build_test_tool_registry())

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="read-1",
                name="read_file",
                input={"path": "example.txt", "offset": 1, "limit": 3},
            ),
            context(tmp_path),
        )
    )

    assert tool.is_read_only(ReadFileInput(path="example.txt")) is True
    assert tool.is_concurrency_safe(ReadFileInput(path="example.txt")) is True
    assert result.error is None
    assert result.data == {
        "path": "example.txt",
        "content": "bcd",
        "offset": 1,
        "complete": False,
    }


def test_read_file_rejects_a_page_larger_than_its_inline_context_bound() -> None:
    with pytest.raises(ValueError):
        ReadFileInput(path="README.md", limit=50_001)


def test_read_file_returns_unchanged_stub_for_an_exact_duplicate(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("abcdef", encoding="utf-8")
    tool = ReadFileTool()
    first_call = ToolUseBlock(
        id="read-1",
        name="read_file",
        input={"path": "README.md", "offset": 1, "limit": 3},
    )
    first = asyncio.run(
        tool.call(ReadFileInput.model_validate(first_call.input), context(tmp_path))
    )
    transcript = (
        RuntimeMessage(role="assistant", content=[first_call]),
        RuntimeMessage(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id=first_call.id,
                    content=first.model_dump(mode="json"),
                )
            ],
        ),
    )
    duplicate_context = apply_tool_result(context(tmp_path), first).model_copy(
        update={"transcript_messages": transcript}
    )

    duplicate = asyncio.run(
        tool.call(
            ReadFileInput(path="README.md", offset=1, limit=3),
            duplicate_context,
        )
    )

    assert duplicate.error is None
    assert duplicate.data["content"].startswith("File unchanged since last exact read")
    assert duplicate.data["offset"] == 1
    assert duplicate.data["complete"] is False
    assert duplicate.state_changes == ()


def test_read_file_does_not_deduplicate_after_the_file_changes(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("first", encoding="utf-8")
    call = ToolUseBlock(id="read-1", name="read_file", input={"path": "README.md"})
    first = asyncio.run(ReadFileTool().call(ReadFileInput(path="README.md"), context(tmp_path)))
    target.write_text("second", encoding="utf-8")
    transcript = (
        RuntimeMessage(role="assistant", content=[call]),
        RuntimeMessage(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id=call.id,
                    content=first.model_dump(mode="json"),
                )
            ],
        ),
    )
    changed_context = apply_tool_result(context(tmp_path), first).model_copy(
        update={"transcript_messages": transcript}
    )

    result = asyncio.run(
        ReadFileTool().call(ReadFileInput(path="README.md"), changed_context)
    )

    assert result.data["content"] == "second"


def test_write_file_requires_permission_and_uses_atomic_write(tmp_path: Path) -> None:
    registry = build_test_tool_registry()
    denied = ToolExecutor(registry)
    call = ToolUseBlock(
        id="write-1",
        name="write_file",
        input={"path": "docs/result.txt", "content": "written"},
    )

    denied_result = asyncio.run(denied.execute(call, context(tmp_path)))
    assert denied_result.error and denied_result.error.code == "PERMISSION_REQUIRED"
    assert not (tmp_path / "docs" / "result.txt").exists()

    async def approve(decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    approved = ToolExecutor(
        registry,
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    approved_result = asyncio.run(approved.execute(call, context(tmp_path)))

    assert approved_result.error is None
    assert approved_result.data == {"path": "docs/result.txt", "chars_written": 7}
    assert (tmp_path / "docs" / "result.txt").read_text(encoding="utf-8") == "written"


def test_file_tools_reject_noncanonical_and_escaping_paths_before_call(tmp_path: Path) -> None:
    executor = ToolExecutor(build_test_tool_registry())

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="read-1", name="read_file", input={"path": "../outside.txt"}),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert "non-canonical" in result.error.message


def test_file_tool_pydantic_contract_rejects_implicit_types(tmp_path: Path) -> None:
    executor = ToolExecutor(build_test_tool_registry())

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="read-1",
                name="read_file",
                input={"path": "example.txt", "limit": "3"},
            ),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "TOOL_INPUT_INVALID"


def test_edit_file_requires_approval_and_replaces_once(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("old old", encoding="utf-8")

    async def approve(decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    registry = build_test_tool_registry()
    executor = ToolExecutor(
        registry,
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    read = asyncio.run(
        executor.execute(
            ToolUseBlock(id="read-before-edit", name="read_file", input={"path": "example.txt"}),
            context(tmp_path),
        )
    )
    edit_context = apply_tool_result(context(tmp_path), read)
    result = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="edit-1",
                name="edit_file",
                input={"path": "example.txt", "old_text": "old", "new_text": "new"},
            ),
            edit_context,
        )
    )

    assert result.error is None
    assert result.data == {"path": "example.txt", "replacements": 1}
    assert target.read_text(encoding="utf-8") == "new old"


def test_glob_returns_structured_paths_and_is_concurrency_safe(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    registry = build_test_tool_registry()
    tool = registry.get("glob")

    result = asyncio.run(
        ToolExecutor(registry).execute(
            ToolUseBlock(id="glob-1", name="glob", input={"pattern": "src/*.py"}),
            context(tmp_path),
        )
    )

    assert tool.is_read_only(GlobInput(pattern="src/*.py")) is True
    assert tool.is_concurrency_safe(GlobInput(pattern="src/*.py")) is True
    assert result.data == {"paths": ["src/a.py", "src/b.py"]}


def test_partial_read_cannot_authorize_an_existing_file_edit(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("abcdef", encoding="utf-8")

    async def approve(_decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    executor = ToolExecutor(
        build_test_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    read = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="partial",
                name="read_file",
                input={"path": "example.txt", "limit": 3},
            ),
            context(tmp_path),
        )
    )
    edit_context = apply_tool_result(context(tmp_path), read)
    edited = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="edit",
                name="edit_file",
                input={"path": "example.txt", "old_text": "abc", "new_text": "xyz"},
            ),
            edit_context,
        )
    )

    assert edited.error and edited.error.code == "FILE_NOT_FULLY_READ"
    assert target.read_text(encoding="utf-8") == "abcdef"


def test_external_change_after_read_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("before", encoding="utf-8")

    async def approve(_decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    executor = ToolExecutor(
        build_test_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    read = asyncio.run(
        executor.execute(
            ToolUseBlock(id="read", name="read_file", input={"path": "example.txt"}),
            context(tmp_path),
        )
    )
    target.write_text("changed externally", encoding="utf-8")
    edit_context = apply_tool_result(context(tmp_path), read)
    edited = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="edit",
                name="edit_file",
                input={"path": "example.txt", "old_text": "before", "new_text": "after"},
            ),
            edit_context,
        )
    )

    assert edited.error and edited.error.code == "FILE_CHANGED_SINCE_READ"
    assert target.read_text(encoding="utf-8") == "changed externally"


def test_new_nested_instruction_blocks_first_write_and_activates_context(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("Use src conventions.", encoding="utf-8")

    async def approve(_decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    executor = ToolExecutor(
        build_test_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    result = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="write",
                name="write_file",
                input={"path": "src/new.py", "content": "value = 1\n"},
            ),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "REPOSITORY_INSTRUCTIONS_DISCOVERED"
    updated = apply_tool_result(context(tmp_path), result)
    assert updated.workspace.instruction_state.active_paths == ("src/AGENTS.md",)
    assert not (nested / "new.py").exists()
