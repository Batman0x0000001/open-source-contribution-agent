from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from osc_agent.runtime.context import ContextPipeline, SessionTranscript
from osc_agent.runtime.instructions import (
    MAX_INSTRUCTION_CHARS,
    RepositoryInstructionResolver,
)
from osc_agent.runtime.models import (
    QueryConfig,
    RepositoryInstructionState,
    RuntimeMessage,
    TextBlock,
    ToolUseContext,
)


def test_root_and_nested_agents_and_claude_files_are_loaded_at_equal_scope(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("root agents", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("root claude", encoding="utf-8")
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested agents", encoding="utf-8")
    (nested / "module.py").write_text("value = 1", encoding="utf-8")
    resolver = RepositoryInstructionResolver()

    state = resolver.activate_for_path(
        tmp_path,
        "src/module.py",
        RepositoryInstructionState(),
    )
    documents = resolver.load(tmp_path, state)

    assert [item.path for item in documents] == [
        "AGENTS.md",
        "CLAUDE.md",
        "src/AGENTS.md",
    ]
    assert documents[0].scope_directory == documents[1].scope_directory == "."


def test_instruction_context_is_reinjected_and_marks_conflict_policy(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("Use focused tests.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Follow repository style.", encoding="utf-8")
    resolver = RepositoryInstructionResolver()
    state = resolver.activate_root(tmp_path)
    transcript = SessionTranscript(
        session_id="instructions",
        messages=[RuntimeMessage(role="user", content=[TextBlock(text="work")])],
    )
    context = ToolUseContext(
        session_id="instructions",
        working_directory=str(tmp_path),
        repository_root=str(tmp_path),
        state_directory=str(tmp_path / "state"),
        instruction_state=state,
    )

    projection = asyncio.run(
        ContextPipeline().project(
            transcript,
            config=QueryConfig(auto_compact_chars=1),
            working_directory=str(tmp_path),
            runtime_context=context,
        )
    )

    assert "Use focused tests." in projection.system_reminder
    assert "Follow repository style." in projection.system_reminder
    assert "task-relevant conflict" in projection.system_reminder
    assert projection.compacted is True


def test_instruction_truncation_and_symlink_escape_are_safe(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "x" * (MAX_INSTRUCTION_CHARS + 100),
        encoding="utf-8",
    )
    resolver = RepositoryInstructionResolver()
    document = resolver.load(tmp_path, resolver.activate_root(tmp_path))[0]
    assert "truncated at 40000" in document.content

    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "nested"
    try:
        link.symlink_to(outside.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="escapes repository"):
        resolver.activate_for_path(tmp_path, "nested/file.py", RepositoryInstructionState())
