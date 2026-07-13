from __future__ import annotations

from types import SimpleNamespace

from osc_agent.agent_loop import TOOLS, agent_loop
from osc_agent.config import Settings
from osc_agent.harness.memory import write_memory_file
from osc_agent.harness.runtime_state import RuntimeState, save_runtime_state
from osc_agent.harness.prompt import (
    PROMPT_SECTIONS,
    PromptContext,
    assemble_system_prompt,
    get_system_prompt,
    update_context,
)
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
        "task",
        "tools",
        "permissions",
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
        messages=[{"role": "user", "content": "Update README docs"}],
        enabled_tools=["bash", "read_file", "load_skill"],
    )

    assert context.repo_root == str(tmp_path.resolve())
    assert context.current_task == "Update README docs"
    assert context.enabled_tools == ["bash", "load_skill", "read_file"]
    assert "README.md" in context.repo_overview
    assert "File tools are limited" in context.permissions
    assert "Docs Preference" in context.memory
    assert context.current_todos[0]["content"] == "Read contribution guide"
    assert context.git_state


def test_assemble_system_prompt_snapshot_contains_expected_sections(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    context = update_context(
        repo_root=tmp_path,
        messages=[{"role": "user", "content": "Fix tests"}],
        enabled_tools=["bash", "compact"],
    )

    prompt = assemble_system_prompt(context)

    assert "You are a coding agent." in prompt
    assert "Repository context:" in prompt
    assert "Current task:\nFix tests" in prompt
    assert "Enabled tools:\nbash, compact" in prompt
    assert "Permission boundaries:" in prompt
    assert "Skills available:" in prompt
    assert "Persistent memory:" in prompt
    assert "Current todos:" in prompt
    assert "Git state:" in prompt
    assert "Do not run git push or open pull requests automatically." in prompt


def test_get_system_prompt_caches_same_context():
    context = PromptContext(
        repo_root="repo",
        current_task="task",
        enabled_tools=["bash"],
        repo_overview="overview",
        permissions="permissions",
        skill_catalog="- docs: Docs",
        suggested_skills=["docs"],
        memory="memory",
        current_todos=[],
        git_state="git",
        runtime_state={"current_goal": "task"},
    )

    first = get_system_prompt(context)
    second = get_system_prompt(context)

    assert first is second


def test_structured_runtime_state_is_injected_independently_of_message_history(tmp_path):
    save_runtime_state(
        tmp_path,
        RuntimeState(
            current_goal="old goal",
            user_constraints=["Do not modify CI"],
            allowed_files=["agent.py"],
            test_results=[{"command": "pytest", "ok": True}],
        ),
    )

    context = update_context(
        repo_root=tmp_path,
        messages=[{"role": "user", "content": "continue implementation"}],
        enabled_tools=["read_file"],
    )
    prompt = assemble_system_prompt(context)

    assert "Authoritative runtime state (never compacted):" in prompt
    assert "Do not modify CI" in prompt
    assert "agent.py" in prompt
    assert "pytest" in prompt


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
    )

    assert response.stop_reason == "end_turn"
    assert len(client.messages.calls) == 2
    assert "say hello" in client.messages.calls[0]["system"]
    assert "tool_result" in client.messages.calls[1]["system"]
    assert {tool["name"] for tool in TOOLS} >= {"bash", "compact", "load_skill"}
