from __future__ import annotations

from types import SimpleNamespace
import json

from osc_agent.agent_loop import _budget_reason, agent_loop
from osc_agent.config import Settings
from osc_agent.harness.contracts import RunMetrics, RunStatus


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
    result = json.loads(messages[2]["content"][0]["content"])
    assert result["ok"] is True
    assert result["summary"] == "ran: echo hello"
    assert result["call_id"] == "toolu_1"


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


class LoopingMessages:
    def __init__(self, *, include_usage=False) -> None:
        self.calls = 0
        self.include_usage = include_usage

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(type="tool_use", name="read_file", id=f"toolu_{self.calls}", input={"path": "a.py"})],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5) if self.include_usage else None,
        )


def test_agent_loop_stops_repeated_actions_with_structured_status(tmp_path):
    settings = Settings(None, None, "test", None, repeat_action_limit=3)
    result = agent_loop(
        [{"role": "user", "content": "loop"}],
        client=SimpleNamespace(messages=LoopingMessages()),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"read_file": lambda path: "same"},
    )

    assert result.status is RunStatus.BLOCKED_NEEDS_USER
    assert "repeated" in result.reason


def test_agent_loop_stops_at_token_budget(tmp_path):
    class TokenMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="done")],
                usage=SimpleNamespace(input_tokens=120, output_tokens=90),
            )

    settings = Settings(None, None, "test", None, max_total_tokens=200)
    result = agent_loop(
        [{"role": "user", "content": "work"}],
        client=SimpleNamespace(messages=TokenMessages()),
        settings=settings,
        repo_root=tmp_path,
    )

    assert result.status is RunStatus.FAILED_BUDGET
    assert result.metrics.total_tokens == 210


def test_agent_loop_stops_at_deadline_before_model_call(tmp_path):
    messages = LoopingMessages()
    settings = Settings(None, None, "test", None, agent_deadline_seconds=0)
    result = agent_loop(
        [{"role": "user", "content": "work"}],
        client=SimpleNamespace(messages=messages),
        settings=settings,
        repo_root=tmp_path,
    )

    assert result.status is RunStatus.FAILED_BUDGET
    assert messages.calls == 0


def test_budget_override_is_per_run_and_per_limit():
    settings = Settings(None, None, "test", None, max_agent_rounds=1, max_total_tokens=10)
    metrics = RunMetrics(model_calls=1, input_tokens=10)
    overrides: set[str] = set()
    prompts: list[str] = []

    def confirm_first(prompt: str) -> bool:
        prompts.append(prompt)
        return len(prompts) == 1

    reason = _budget_reason(metrics, settings, confirm=confirm_first, overrides=overrides)

    assert reason == "maximum token budget reached (10)"
    assert overrides == {"rounds"}
    assert _budget_reason(metrics, settings) == "maximum model rounds reached (1)"


def test_agent_loop_stops_after_consecutive_tool_failures(tmp_path):
    class FailingMessages:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="read_file",
                        id=f"toolu_{self.calls}",
                        input={"path": f"missing_{self.calls}.py"},
                    )
                ],
            )

    settings = Settings(None, None, "test", None, consecutive_failure_limit=3)
    result = agent_loop(
        [{"role": "user", "content": "read missing files"}],
        client=SimpleNamespace(messages=FailingMessages()),
        settings=settings,
        repo_root=tmp_path,
        tool_handlers={"read_file": lambda path: f"Error: not found: {path}"},
    )

    assert result.status is RunStatus.FAILED_TOOL
    assert result.metrics.tool_failures == 3
