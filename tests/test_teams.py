from __future__ import annotations

import time
from copy import deepcopy
from types import SimpleNamespace

from osc_agent.agent_loop import TOOLS, agent_loop, build_tool_handlers
from osc_agent.config import Settings
from osc_agent.harness.teams import MessageBus, check_inbox, collect_team_notifications, spawn_teammate


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


class FinishImmediatelyMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="Reviewed diff risks.")],
        )


class FakeClient:
    def __init__(self, messages) -> None:
        self.messages = messages


def test_message_bus_sends_and_consumes_inbox(tmp_path):
    bus = MessageBus(tmp_path)

    result = bus.send("lead", "alice", "please review", "task_assignment", {"file": "README.md"})
    messages = bus.read_inbox("alice")

    assert result == "Sent message to alice"
    assert messages[0]["from_agent"] == "lead"
    assert messages[0]["content"] == "please review"
    assert bus.read_inbox("alice") == []


def test_check_inbox_formats_and_consumes_lead_messages(tmp_path):
    MessageBus(tmp_path).send("alice", "lead", "done", "result")

    output = check_inbox(repo_root=tmp_path)

    assert "<teammate-message>" in output
    assert "<from>alice</from>" in output
    assert check_inbox(repo_root=tmp_path) == "(inbox empty)"


def test_spawn_teammate_sends_final_result_to_lead(tmp_path):
    fake_messages = FinishImmediatelyMessages()

    result = spawn_teammate(
        name="alice",
        role="reviewer",
        prompt="Review the diff.",
        repo_root=tmp_path,
        client=FakeClient(fake_messages),
        settings=_settings(),
    )

    for _ in range(50):
        inbox = collect_team_notifications(tmp_path)
        if inbox:
            break
        time.sleep(0.02)

    assert result == "Spawned teammate alice as reviewer"
    assert "Teammate alice (reviewer) result:" in inbox[0]
    assert "Reviewed diff risks." in inbox[0]
    assert "write_file" not in {tool["name"] for tool in fake_messages.calls[0]["tools"]}


def test_spawn_teammate_can_explicitly_allow_write_tool(tmp_path):
    fake_messages = FinishImmediatelyMessages()

    spawn_teammate(
        name="docs",
        role="doc_writer",
        prompt="Update docs.",
        repo_root=tmp_path,
        client=FakeClient(fake_messages),
        settings=_settings(),
        allow_write=True,
    )

    for _ in range(50):
        if collect_team_notifications(tmp_path):
            break
        time.sleep(0.02)

    assert "write_file" in {tool["name"] for tool in fake_messages.calls[0]["tools"]}


def test_agent_loop_registers_team_tools(tmp_path):
    tool_names = {tool["name"] for tool in TOOLS}
    handlers = build_tool_handlers(tmp_path, client=FakeClient(FinishImmediatelyMessages()), settings=_settings())

    assert {"spawn_teammate", "send_message", "check_inbox"} <= tool_names
    assert handlers["send_message"](to_agent="lead", content="hello") == "Sent message to lead"
    assert "hello" in handlers["check_inbox"]()


class LeadInboxMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ack")])


def test_agent_loop_injects_lead_inbox_messages(tmp_path):
    MessageBus(tmp_path).send("alice", "lead", "tests passed", "result")
    fake_messages = LeadInboxMessages()
    messages = [{"role": "user", "content": "continue"}]

    agent_loop(
        messages,
        client=FakeClient(fake_messages),
        settings=_settings(),
        repo_root=tmp_path,
    )

    assert messages[1]["role"] == "user"
    assert "<from>alice</from>" in messages[1]["content"][0]["text"]
    assert "tests passed" in fake_messages.calls[0]["messages"][1]["content"][0]["text"]
