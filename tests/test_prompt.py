from __future__ import annotations

from types import SimpleNamespace

import pytest

from osc_agent.agent_loop import TOOLS, agent_loop
from osc_agent.config import Settings
from osc_agent.harness.capabilities import AgentCapabilityScope, AgentExecutionStage
from osc_agent.harness.memory import write_memory_file
from osc_agent.harness.runtime_state import RuntimeState, save_runtime_state
from osc_agent.harness.prompt import (
    PROMPT_SECTIONS,
    SYSTEM_PROMPT_CHAR_LIMIT,
    PromptContext,
    assemble_system_prompt,
    update_context,
)
from osc_agent.harness.repository_boundary import repository_boundary_summary
from osc_agent.harness.risk import DENIED_SHELL_PATTERNS, risk_policy_summary
from osc_agent.harness.todo import todo_write


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


def test_prompt_sections_are_defined():
    assert set(PROMPT_SECTIONS) == {
        "identity",
        "repo",
        "objective",
        "instruction",
        "effective_boundary",
        "repository_boundary",
        "risk",
        "skills",
        "memory",
        "current_todos",
        "git_state",
        "runtime_state",
    }


def test_update_context_uses_real_repo_tools_memory_and_todos(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    write_memory_file(
        tmp_path,
        name="Docs Preference",
        mem_type="reference",
        description="README changes should be additive.",
        body="Append README updates instead of replacing existing content.",
    )
    todo_write([{"content": "Read contribution guide", "status": "pending"}], repo_root=tmp_path)

    context = update_context(
        repo_root=tmp_path,
        objective="Improve project documentation",
        current_instruction="Update README docs",
        enabled_tools=["bash", "read_file", "load_skill"],
        capabilities=AgentCapabilityScope.unrestricted(),
    )

    assert context.objective == "Improve project documentation"
    assert context.current_instruction == "Update README docs"
    assert context.enabled_tools == ["bash", "load_skill", "read_file"]
    assert "README.md" in context.repo_overview
    assert context.capabilities.stage is AgentExecutionStage.INTERACTIVE
    assert all(pattern in context.risk_controls for pattern in DENIED_SHELL_PATTERNS)
    assert "Docs Preference" in context.memory
    assert context.current_todos[0]["content"] == "Read contribution guide"
    assert context.git_state


def test_assemble_system_prompt_snapshot_contains_expected_sections(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    context = update_context(
        repo_root=tmp_path,
        objective="Fix the regression",
        current_instruction="Fix tests",
        enabled_tools=["bash", "compact"],
        capabilities=AgentCapabilityScope.unrestricted(),
    )

    prompt = assemble_system_prompt(context)

    assert "You are a coding agent." in prompt
    assert "Repository context:" in prompt
    assert "Top-level objective:\nFix the regression" in prompt
    assert "Current execution instruction:\nFix tests" in prompt
    assert "Effective execution boundary:" in prompt
    assert '"effective_tools": [' in prompt
    assert '"bash"' in prompt
    assert '"compact"' in prompt
    assert "Repository boundary:" in prompt
    assert "Risk controls:" in prompt
    assert "Skills available:" not in prompt
    assert "Persistent memory:" in prompt
    assert "Current todos:" in prompt
    assert "Git state:" in prompt
    assert "git push" in prompt
    assert "explicit approval" in prompt
    assert "repository and risk controls remain active" in prompt


def test_prompt_uses_structured_stage_and_capability_paths(tmp_path):
    capabilities = AgentCapabilityScope(
        stage="repair",
        allowed_tools=frozenset({"read_file", "edit_file"}),
        writable_paths=("osc_agent/*.py",),
        forbidden_paths=("tests/test_regression.py",),
    )
    context = update_context(
        repo_root=tmp_path,
        objective="Fix the regression",
        current_instruction="repair the failed implementation",
        enabled_tools=["read_file", "edit_file"],
        capabilities=capabilities,
    )

    prompt = assemble_system_prompt(context)

    assert '"stage": "repair"' in prompt
    assert "osc_agent/*.py" in prompt
    assert "tests/test_regression.py" in prompt
    assert "smallest repair" in prompt


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("understanding", "READY_TO_EDIT"),
        ("reproduce", "smallest regression test"),
        ("edit", "modify only writable_paths"),
        ("repair", "smallest repair"),
        ("verification", "verification evidence"),
    ],
)
def test_prompt_covers_each_structured_contribution_stage(tmp_path, stage, expected):
    context = update_context(
        repo_root=tmp_path,
        objective="Complete the selected contribution",
        current_instruction="stage instructions are independent of task wording",
        enabled_tools=["read_file"],
        capabilities=AgentCapabilityScope(stage=stage, allowed_tools=frozenset({"read_file"})),
        session_id=f"stage-{stage}",
    )

    assert expected in assemble_system_prompt(context)


def test_structured_runtime_state_is_injected_independently_of_message_history(tmp_path):
    save_runtime_state(
        tmp_path,
        RuntimeState(
            objective="old goal",
            current_instruction="old instruction",
            scope_exclusions=["Do not modify CI"],
            allowed_files=["agent.py"],
            recent_test_observations=[{"command": "pytest", "ok": True}],
        ),
    )

    context = update_context(
        repo_root=tmp_path,
        objective="new goal",
        current_instruction="continue implementation",
        enabled_tools=["read_file"],
        capabilities=AgentCapabilityScope.unrestricted(),
    )
    prompt = assemble_system_prompt(context)

    assert "Authoritative runtime state (never compacted):" in prompt
    assert "Do not modify CI" in prompt
    assert '"objective"' not in prompt
    assert '"current_instruction"' not in prompt
    assert '"allowed_files"' not in prompt
    assert '"modified_files"' not in prompt
    assert "pytest" in prompt


def test_prompt_only_mentions_optional_tools_when_enabled(tmp_path):
    without_optional_tools = update_context(
        repo_root=tmp_path,
        objective="Inspect the repository",
        current_instruction="inspect",
        enabled_tools=["read_file"],
        capabilities=AgentCapabilityScope.unrestricted(),
    )
    with_optional_tools = update_context(
        repo_root=tmp_path,
        objective="Implement the contribution",
        current_instruction="implement",
        enabled_tools=["read_file", "load_skill", "todo_write"],
        capabilities=AgentCapabilityScope.unrestricted(),
        session_id="with-optional-tools",
    )

    basic_prompt = assemble_system_prompt(without_optional_tools)
    extended_prompt = assemble_system_prompt(with_optional_tools)

    assert "Use load_skill(name)" not in basic_prompt
    assert "call todo_write" not in basic_prompt
    assert "Use load_skill(name)" in extended_prompt
    assert "call todo_write" in extended_prompt


def test_update_context_rejects_tools_outside_capability_scope(tmp_path):
    capabilities = AgentCapabilityScope(
        stage=AgentExecutionStage.UNDERSTANDING,
        allowed_tools=frozenset({"read_file"}),
    )

    with pytest.raises(ValueError, match="enabled_tools exceed the capability scope: bash"):
        update_context(
            repo_root=tmp_path,
            objective="Inspect the repository",
            current_instruction="inspect",
            enabled_tools=["read_file", "bash"],
            capabilities=capabilities,
        )


def test_prompt_has_total_budget_and_tolerates_malformed_todos():
    context = PromptContext(
        objective="o" * 20_000,
        current_instruction="t" * 20_000,
        enabled_tools=["read_file"],
        repo_overview="r" * 20_000,
        capabilities=AgentCapabilityScope.unrestricted(),
        boundary_controls=repository_boundary_summary(),
        risk_controls=risk_policy_summary(),
        skill_catalog="",
        suggested_skills=[],
        memory="m" * 20_000,
        current_todos=[{}, "broken"],  # type: ignore[list-item]
        git_state="g" * 20_000,
        runtime_state={"verification_results": ["x" * 20_000]},
    )

    prompt = assemble_system_prompt(context)

    assert len(prompt) <= SYSTEM_PROMPT_CHAR_LIMIT
    assert "...[truncated]" in prompt
    assert "Effective execution boundary:" in prompt
    assert "Repository boundary:" in prompt
    assert "Risk controls:" in prompt
    assert "unknown: (missing content)" in prompt
    assert "unknown: (malformed todo)" in prompt


def test_agent_execution_stage_rejects_unknown_values():
    with pytest.raises(ValueError):
        AgentCapabilityScope(stage="typo")


class TwoTurnMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="bash",
                        id="toolu_1",
                        input={"command": "echo hello"},
                    )
                ],
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="done")],
        )


class FakeClient:
    def __init__(self) -> None:
        self.messages = TwoTurnMessages()


def test_agent_loop_reassembles_system_prompt_each_round(tmp_path):
    messages = [{"role": "user", "content": "say hello"}]
    client = FakeClient()

    response = agent_loop(
        messages,
        client=client,
        settings=_settings(),
        repo_root=tmp_path,
        tool_handlers={"bash": lambda command: f"ran: {command}"},
        objective="Respond to the greeting request",
    )

    assert response.stop_reason == "end_turn"
    assert len(client.messages.calls) == 2
    assert "Top-level objective:\nRespond to the greeting request" in client.messages.calls[0]["system"]
    assert "Current execution instruction:\nsay hello" in client.messages.calls[0]["system"]
    assert "say hello" in client.messages.calls[0]["system"]
    assert "say hello" in client.messages.calls[1]["system"]
    assert "tool_result" not in client.messages.calls[1]["system"]
    assert {tool["name"] for tool in TOOLS} >= {"bash", "compact", "load_skill"}
