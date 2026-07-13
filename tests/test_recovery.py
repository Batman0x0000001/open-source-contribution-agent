from __future__ import annotations

import sys
from types import SimpleNamespace

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings
from osc_agent.harness.recovery import (
    CONTINUATION_PROMPT,
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    RecoveryState,
    classify_model_error,
    is_overloaded_error,
    is_prompt_too_long_error,
    is_rate_limit_error,
    with_retry,
)
from osc_agent.tools.shell import run_bash


def _settings(fallback_model_id: str | None = None) -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="primary-model",
        fallback_model_id=fallback_model_id,
    )


def test_error_classifiers():
    assert is_prompt_too_long_error(RuntimeError("prompt_too_long"))
    assert is_rate_limit_error(RuntimeError("429 rate limit"))
    assert is_overloaded_error(RuntimeError("529 overloaded"))
    assert classify_model_error(RuntimeError("401 authentication failed")) == "AUTHENTICATION_ERROR"
    assert classify_model_error(RuntimeError("connection timeout")) == "NETWORK_ERROR"


def test_with_retry_handles_rate_limit_without_crashing(tmp_path):
    calls = []
    sleeps = []

    def create(model_id):
        calls.append(model_id)
        if len(calls) < 3:
            raise RuntimeError("429 rate limit")
        return "ok"

    result = with_retry(
        create,
        state=RecoveryState(current_model="primary"),
        repo_root=tmp_path,
        sleep_fn=sleeps.append,
    )

    assert result == "ok"
    assert calls == ["primary", "primary", "primary"]
    assert len(sleeps) == 2


def test_with_retry_switches_to_fallback_after_repeated_529(tmp_path):
    calls = []

    def create(model_id):
        calls.append(model_id)
        if len(calls) <= 3:
            raise RuntimeError("529 overloaded")
        return "ok"

    state = RecoveryState(current_model="primary", fallback_model_id="fallback")
    result = with_retry(create, state=state, repo_root=tmp_path, sleep_fn=lambda _delay: None)

    assert result == "ok"
    assert calls[-1] == "fallback"
    assert state.current_model == "fallback"


class MaxTokensMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(stop_reason="max_tokens", content=[SimpleNamespace(type="text", text="cut")])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")])


class FakeClient:
    def __init__(self, messages) -> None:
        self.messages = messages


def test_agent_loop_escalates_max_tokens_before_appending_truncated_output(tmp_path):
    messages = [{"role": "user", "content": "write a long answer"}]
    fake_messages = MaxTokensMessages()

    response = agent_loop(
        messages,
        client=FakeClient(fake_messages),
        settings=_settings(),
        repo_root=tmp_path,
    )

    assert response.stop_reason == "end_turn"
    assert fake_messages.calls[0]["max_tokens"] == DEFAULT_MAX_TOKENS
    assert fake_messages.calls[1]["max_tokens"] == ESCALATED_MAX_TOKENS
    assert all("cut" not in str(message.get("content")) for message in messages)


class RepeatedMaxTokensMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= 2:
            return SimpleNamespace(stop_reason="max_tokens", content=[SimpleNamespace(type="text", text=f"cut-{self.calls}")])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")])


def test_agent_loop_adds_continuation_prompt_after_escalated_truncation(tmp_path):
    messages = [{"role": "user", "content": "continue long output"}]

    response = agent_loop(
        messages,
        client=FakeClient(RepeatedMaxTokensMessages()),
        settings=_settings(),
        repo_root=tmp_path,
    )

    assert response.stop_reason == "end_turn"
    assert any(message.get("content") == CONTINUATION_PROMPT for message in messages)


def test_run_bash_timeout_is_structured_error(tmp_path):
    command = f'{sys.executable} -c "import time; time.sleep(2)"'

    output = run_bash(command, repo_root=tmp_path, timeout_seconds=0.1)

    assert output.startswith("Error:")
    assert '"kind": "timeout"' in output
    assert "timed out" in output


def test_run_bash_test_failure_includes_recovery_guidance(tmp_path):
    command = f'{sys.executable} -m pytest missing_test_file.py'

    output = run_bash(command, repo_root=tmp_path)

    assert '"kind": "nonzero_exit"' in output
    assert "Recovery guidance: tests failed" in output
    assert "update todo" in output
