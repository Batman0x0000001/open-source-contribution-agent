from __future__ import annotations

import json
import subprocess

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from osc_agent.agent_loop import TOOLS, agent_loop
from osc_agent.config import Settings
from osc_agent.harness.subagent import SUBAGENT_TOOLS, run_read_only_bash, spawn_subagent
from osc_agent.harness.trace import trace_path


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


class SubagentReadFileMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="read_file",
                        id="toolu_sub_1",
                        input={"path": "README.md"},
                    )
                ],
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Findings: README is present.")],
        )


class FakeClient:
    def __init__(self, messages) -> None:
        self.messages = messages


def test_spawn_subagent_uses_fresh_context_and_returns_summary_only(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    messages = SubagentReadFileMessages()

    result = spawn_subagent(
        "Map docs",
        "repo_mapper",
        client=FakeClient(messages),
        settings=_settings(),
        repo_root=tmp_path,
    )

    assert "Findings: README is present." in result
    assert "toolu_sub_1" not in result
    assert messages.calls[0]["messages"] == [{"role": "user", "content": "Map docs"}]
    assert {tool["name"] for tool in messages.calls[0]["tools"]} == {
        "bash",
        "read_file",
        "glob",
        "git_status",
        "inspect_repo",
    }


def test_spawn_subagent_rejects_unknown_role(tmp_path):
    with pytest.raises(ValueError, match="role must be one of"):
        spawn_subagent(
            "Analyze",
            "writer",
            client=FakeClient(SubagentReadFileMessages()),
            settings=_settings(),
            repo_root=tmp_path,
        )


class SubagentBashMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="bash",
                        id="toolu_sub_1",
                        input={"command": "echo writes are not allowed"},
                    )
                ],
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Done")],
        )


def test_subagent_bash_is_read_only(tmp_path):
    messages = SubagentBashMessages()

    spawn_subagent(
        "Try command",
        "test_analyzer",
        client=FakeClient(messages),
        settings=_settings(),
        repo_root=tmp_path,
    )

    tool_result = messages.calls[1]["messages"][-1]["content"][0]["content"]
    assert tool_result == "Permission denied: subagent bash is read-only"


@pytest.mark.parametrize(
    "command",
    [
        "git status & echo bypassed",
        "git status > status.txt",
        "rg --pre malicious pattern",
        "git diff --output=diff.txt",
        "rg pattern ../outside",
    ],
)
def test_read_only_bash_rejects_shell_and_write_bypasses(tmp_path, command):
    result = run_read_only_bash(command, repo_root=tmp_path)

    assert result.startswith("Permission denied:")
    assert list(tmp_path.iterdir()) == []


def test_read_only_bash_allows_structured_git_status(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    result = run_read_only_bash("git status --short", repo_root=tmp_path)

    assert not result.startswith(("Error:", "Permission denied:"))


class MainAndSubagentMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="subagent",
                        id="toolu_main_1",
                        input={"description": "Map repo", "role": "repo_mapper"},
                    )
                ],
            )
        if len(self.calls) == 2:
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="Summary only.")],
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Parent done.")],
        )


def test_agent_loop_exposes_subagent_and_parent_receives_only_summary(tmp_path):
    messages = [{"role": "user", "content": "delegate repo map"}]
    fake_messages = MainAndSubagentMessages()

    response = agent_loop(
        messages,
        client=FakeClient(fake_messages),
        settings=_settings(),
        repo_root=tmp_path,
    )

    assert response.stop_reason == "end_turn"
    tool_result = messages[2]["content"][0]["content"]
    payload = json.loads(tool_result)
    assert "Summary only." in payload["summary"]
    assert payload["call_id"] == "toolu_main_1"
    assert {tool["name"] for tool in TOOLS} >= {"subagent", "todo_write"}
    assert "subagent" not in {tool["name"] for tool in SUBAGENT_TOOLS}


def test_subagent_trace_events_include_agent_marker(tmp_path):
    spawn_subagent(
        "Map docs",
        "doc_reviewer",
        client=FakeClient(SubagentReadFileMessages()),
        settings=_settings(),
        repo_root=tmp_path,
    )

    records = [
        json.loads(line)
        for line in trace_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]

    assert any(record["event"] == "subagent_start" and record["agent"] == "subagent" for record in records)
    assert any(record["event"] == "subagent_tool_use" and record["agent"] == "subagent" for record in records)
    assert any(record["event"] == "subagent_stop" and record["agent"] == "subagent" for record in records)
