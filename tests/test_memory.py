from __future__ import annotations

import pytest

from osc_agent.harness.memory import (
    ensure_memory_store,
    extract_repo_memories,
    list_memory_files,
    load_memory_index,
    load_relevant_memories,
    memory_index_path,
    memory_prompt,
    select_relevant_memories,
    write_memory_file,
)
from osc_agent.harness.capabilities import AgentCapabilityScope
from osc_agent.harness.prompt import assemble_system_prompt, update_context


def test_ensure_memory_store_creates_readable_index(tmp_path):
    ensure_memory_store(tmp_path)

    index = memory_index_path(tmp_path)
    assert index.exists()
    assert index.read_text(encoding="utf-8").startswith("# Memory Index")


def test_write_memory_file_updates_markdown_index(tmp_path):
    path = write_memory_file(
        tmp_path,
        name="Test Command",
        mem_type="project",
        description="Run pytest for this project.",
        body="Use `python -m pytest tests`.",
    )

    assert path.name == "test-command.md"
    assert "type: project" in path.read_text(encoding="utf-8")
    index = load_memory_index(tmp_path)
    assert "[Test Command](test-command.md)" in index


def test_write_memory_file_rejects_sensitive_text(tmp_path):
    with pytest.raises(ValueError, match="sensitive"):
        write_memory_file(
            tmp_path,
            name="API key",
            mem_type="user",
            description="secret",
            body="token abc123",
        )


def test_extract_repo_memories_saves_reusable_project_facts(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "CONTRIBUTING.md").write_text("Run tests first.", encoding="utf-8")

    written = extract_repo_memories(tmp_path)
    entries = list_memory_files(tmp_path)

    assert written
    assert {entry.name for entry in entries} >= {"python-test-command", "contribution-guide"}
    assert "python -m pytest tests" in load_relevant_memories(tmp_path, "python tests")


def test_select_and_load_relevant_memories_with_limit(tmp_path):
    write_memory_file(
        tmp_path,
        name="Python Tests",
        mem_type="project",
        description="Use pytest for Python changes.",
        body="Run `python -m pytest tests` before reporting done.",
    )
    write_memory_file(
        tmp_path,
        name="Docs Style",
        mem_type="reference",
        description="README updates should be additive.",
        body="Append README notes instead of replacing existing text.",
    )

    selected = select_relevant_memories(tmp_path, "How should I test python changes?")
    loaded = load_relevant_memories(tmp_path, "python test", limit_chars=80)

    assert selected[0].name == "Python Tests"
    assert len(loaded) <= 80
    assert "Python Tests" in loaded


def test_memory_prompt_is_limited_and_injected_into_system_prompt(tmp_path):
    write_memory_file(
        tmp_path,
        name="Contribution Guide",
        mem_type="reference",
        description="Read CONTRIBUTING before edits.",
        body="Always read `CONTRIBUTING.md` before changing code.",
    )

    text = memory_prompt(tmp_path, query="contribution", limit_chars=120)
    context = update_context(
        repo_root=tmp_path,
        objective="contribution",
        current_instruction="review contribution memory",
        enabled_tools=[],
        capabilities=AgentCapabilityScope.unrestricted(),
    )
    prompt = assemble_system_prompt(context)

    assert len(text) <= 120
    assert "Persistent memory:" in prompt
    assert "Contribution Guide" in prompt
