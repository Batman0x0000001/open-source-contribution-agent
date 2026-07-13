from __future__ import annotations

import json

from pathlib import Path
from types import SimpleNamespace

import pytest

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings
from osc_agent.harness.hooks import HookRegistry
from osc_agent.harness.permissions import check_shell_command, safe_repo_path
from osc_agent.harness.trace import trace_path
from osc_agent.tools.files import edit_file, read_file, write_file
from osc_agent.tools.shell import run_bash


def test_safe_repo_path_blocks_path_escape(tmp_path):
    with pytest.raises(ValueError):
        safe_repo_path(tmp_path, "../outside.txt")


def test_file_tools_reject_path_escape(tmp_path):
    assert "path escapes repository" in read_file(repo_root=tmp_path, path="../outside.txt")
    assert "path escapes repository" in write_file(
        repo_root=tmp_path,
        path="../outside.txt",
        content="nope",
    )


def test_edit_file_rejects_path_escape(tmp_path):
    assert "path escapes repository" in edit_file(
        repo_root=tmp_path,
        path="../outside.txt",
        old_text="a",
        new_text="b",
    )


def test_dangerous_shell_command_is_denied(tmp_path):
    output = run_bash("git push origin main", repo_root=tmp_path)

    assert output.startswith("Permission denied:")


def test_suspicious_shell_command_requires_confirmation():
    decision = check_shell_command("pip install some-package")

    assert decision.action == "ask"
    assert "explicit confirmation" in decision.reason


def test_large_file_write_requires_confirmation(tmp_path):
    output = write_file(repo_root=tmp_path, path="large.txt", content="x" * 500_001)

    assert output.startswith("Permission required:")


class FakeMessages:
    def __init__(self, tool_name: str, tool_input: dict[str, object]) -> None:
        self.calls = 0
        self.tool_name = tool_name
        self.tool_input = tool_input

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name=self.tool_name,
                        id="toolu_1",
                        input=self.tool_input,
                    )
                ],
            )
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")])


class FakeClient:
    def __init__(self, tool_name: str, tool_input: dict[str, object]) -> None:
        self.messages = FakeMessages(tool_name, tool_input)


def test_pre_tool_hook_blocks_dangerous_command_before_handler(tmp_path):
    messages = [{"role": "user", "content": "push"}]
    settings = Settings(None, None, "test-model", None)

    agent_loop(
        messages,
        client=FakeClient("bash", {"command": "git push origin main"}),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"bash": lambda command: "should not run"},
    )

    tool_results = messages[2]["content"]
    assert isinstance(tool_results, list)
    result = json.loads(tool_results[0]["content"])
    assert result["error_code"] == "PERMISSION_DENIED"
    records = [json.loads(line) for line in trace_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    stop = next(record for record in records if record["event"] == "stop_summary")
    assert stop["failed_count"] == 1


def test_custom_hooks_do_not_replace_default_permission_hook(tmp_path):
    messages = [{"role": "user", "content": "push"}]
    settings = Settings(None, None, "test-model", None)
    custom_hooks = HookRegistry()
    custom_hooks.register("PreToolUse", lambda context, payload: None)

    agent_loop(
        messages,
        client=FakeClient("bash", {"command": "git push origin main"}),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"bash": lambda command: "should not run"},
        hooks=custom_hooks,
    )

    tool_results = messages[2]["content"]
    assert isinstance(tool_results, list)
    result = json.loads(tool_results[0]["content"])
    assert result["error_code"] == "PERMISSION_DENIED"


def test_post_tool_hook_writes_trace_and_stop_summary(tmp_path):
    messages = [{"role": "user", "content": "read"}]
    settings = Settings(None, None, "test-model", None)

    agent_loop(
        messages,
        client=FakeClient("read_file", {"path": "README.md"}),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"read_file": lambda path: "hello"},
    )

    trace_text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert '"event": "permission_decision"' in trace_text
    assert '"event": "tool_use"' in trace_text
    assert '"event": "stop_summary"' in trace_text


def test_ask_permission_runs_handler_when_user_confirms(tmp_path):
    messages = [{"role": "user", "content": "install"}]
    settings = Settings(None, None, "test-model", None)

    agent_loop(
        messages,
        client=FakeClient("bash", {"command": "pip install demo"}),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"bash": lambda command: f"approved: {command}"},
        confirm=lambda prompt: True,
    )

    tool_results = messages[2]["content"]
    assert isinstance(tool_results, list)
    assert json.loads(tool_results[0]["content"])["summary"] == "approved: pip install demo"
    trace_text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert '"event": "permission_confirmation"' in trace_text
    assert '"approved": true' in trace_text


def test_ask_permission_blocks_handler_when_user_rejects(tmp_path):
    messages = [{"role": "user", "content": "install"}]
    settings = Settings(None, None, "test-model", None)

    agent_loop(
        messages,
        client=FakeClient("bash", {"command": "pip install demo"}),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"bash": lambda command: "should not run"},
        confirm=lambda prompt: False,
    )

    tool_results = messages[2]["content"]
    assert isinstance(tool_results, list)
    result = json.loads(tool_results[0]["content"])
    assert result["error_code"] == "PERMISSION_REQUIRED"
    trace_text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert '"approved": false' in trace_text
