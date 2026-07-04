from __future__ import annotations

from types import SimpleNamespace

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings


class FakeMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
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
        self.messages = FakeMessages()


def test_agent_loop_appends_tool_result_and_stops(tmp_path):
    messages = [{"role": "user", "content": "say hello"}]
    settings = Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )

    response = agent_loop(
        messages,
        client=FakeClient(),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"bash": lambda command: f"ran: {command}"},
    )

    assert response.stop_reason == "end_turn"
    assert messages[1]["role"] == "assistant"
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "ran: echo hello",
            }
        ],
    }


def test_agent_loop_starts_background_bash(tmp_path):
    messages = [{"role": "user", "content": "run slow test"}]
    settings = Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )

    class BackgroundMessages(FakeMessages):
        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    stop_reason="tool_use",
                    content=[
                        SimpleNamespace(
                            type="tool_use",
                            name="bash",
                            id="toolu_bg",
                            input={"command": "pytest", "run_in_background": True},
                        )
                    ],
                )
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")])

    class BackgroundClient:
        def __init__(self) -> None:
            self.messages = BackgroundMessages()

    response = agent_loop(
        messages,
        client=BackgroundClient(),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"bash": lambda command: f"ran: {command}"},
    )

    assert response.stop_reason == "end_turn"
    assert "Background task bg_" in messages[2]["content"][0]["content"]
