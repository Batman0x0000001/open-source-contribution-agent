from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
import sys
from types import SimpleNamespace

import pytest

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings
from osc_agent.harness.recovery import (
    CONTINUATION_PROMPT,
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    ModelRequestRecoveryState,
    classify_model_error,
    is_overloaded_error,
    is_prompt_too_long_error,
    is_rate_limit_error,
    with_retry,
)
from osc_agent.harness.trace import trace_path
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


class StructuredError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.type = error_type
        self.request_id = "req_test"
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


def test_error_classifier_prefers_structured_status_and_type():
    assert classify_model_error(StructuredError("busy", status_code=529)) == "OVERLOADED"
    assert (
        classify_model_error(StructuredError("context length mentioned", status_code=529))
        == "OVERLOADED"
    )
    assert (
        classify_model_error(
            StructuredError("limited", status_code=500, error_type="rate_limit_error")
        )
        == "RATE_LIMITED"
    )
    assert classify_model_error(StructuredError("denied", status_code=403)) == "AUTHENTICATION_ERROR"


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
        state=ModelRequestRecoveryState(current_model="primary"),
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

    state = ModelRequestRecoveryState(current_model="primary", fallback_model_id="fallback")
    result = with_retry(create, state=state, repo_root=tmp_path, sleep_fn=lambda _delay: None)

    assert result == "ok"
    assert calls[-1] == "fallback"
    assert state.current_model == "fallback"
    assert state.fallback_switches == 1


def test_non_529_retryable_error_breaks_consecutive_overload_count(tmp_path):
    errors = [
        RuntimeError("529 overloaded"),
        RuntimeError("429 rate limit"),
        RuntimeError("529 overloaded"),
        RuntimeError("529 overloaded"),
    ]
    calls: list[str] = []

    def create(model_id):
        calls.append(model_id)
        if errors:
            raise errors.pop(0)
        return "ok"

    state = ModelRequestRecoveryState(current_model="primary", fallback_model_id="fallback")
    result = with_retry(create, state=state, repo_root=tmp_path, sleep_fn=lambda _delay: None)

    assert result == "ok"
    assert calls == ["primary"] * 5
    assert state.fallback_switches == 0


def test_fallback_is_not_repeated_when_fallback_model_is_overloaded(tmp_path):
    calls: list[str] = []

    def create(model_id):
        calls.append(model_id)
        if len(calls) <= 6:
            raise RuntimeError("529 overloaded")
        return "ok"

    state = ModelRequestRecoveryState(current_model="primary", fallback_model_id="fallback")
    result = with_retry(
        create,
        state=state,
        repo_root=tmp_path,
        sleep_fn=lambda _delay: None,
        max_retries_per_request=7,
    )

    events = [json.loads(line) for line in trace_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    assert result == "ok"
    assert state.fallback_switches == 1
    assert sum(event["event"] == "model_fallback" for event in events) == 1


def test_each_logical_request_has_an_independent_retry_budget(tmp_path):
    first_calls = 0

    def first(_model_id):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            raise RuntimeError("429 rate limit")
        return "ok"

    state = ModelRequestRecoveryState(current_model="primary")
    assert with_retry(
        first,
        state=state,
        repo_root=tmp_path,
        sleep_fn=lambda _delay: None,
        max_retries_per_request=1,
    ) == "ok"

    second_calls = 0

    def second(_model_id):
        nonlocal second_calls
        second_calls += 1
        if second_calls == 1:
            raise RuntimeError("429 rate limit")
        return "ok"

    assert with_retry(
        second,
        state=state,
        repo_root=tmp_path,
        sleep_fn=lambda _delay: None,
        max_retries_per_request=1,
    ) == "ok"

    assert state.logical_request_count == 2
    assert state.total_attempts == 4
    assert state.retry_count == 2


def test_model_error_trace_is_sanitized_bounded_and_counts_attempts(tmp_path):
    calls = 0

    def create(_model_id):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("429 rate limit Authorization: Bearer top-secret " + "x" * 2_000)
        return "ok"

    state = ModelRequestRecoveryState(current_model="primary")
    assert with_retry(create, state=state, repo_root=tmp_path, sleep_fn=lambda _delay: None) == "ok"

    events = [json.loads(line) for line in trace_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    retry_events = [event for event in events if event["event"] == "model_retry"]
    finished = [event for event in events if event["event"] == "model_request_finished"][-1]
    assert "top-secret" not in json.dumps(events)
    assert all(len(event["error"]) <= 1_000 for event in retry_events)
    assert finished["request_attempts"] == 3
    assert finished["request_retries"] == 2
    assert finished["total_attempts"] == 3
    assert finished["total_retries"] == 2


def test_retry_after_supports_http_date(tmp_path):
    calls = 0
    sleeps: list[float] = []
    retry_at = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=2), usegmt=True)

    def create(_model_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StructuredError("limited", status_code=429, headers={"Retry-After": retry_at})
        return "ok"

    assert with_retry(
        create,
        state=ModelRequestRecoveryState(current_model="primary"),
        repo_root=tmp_path,
        sleep_fn=sleeps.append,
    ) == "ok"
    assert len(sleeps) == 1
    assert 0 <= sleeps[0] <= 2


def test_retry_after_above_limit_fails_without_sleeping(tmp_path):
    sleeps: list[float] = []
    error = StructuredError("limited", status_code=429, headers={"Retry-After": "120"})

    with pytest.raises(StructuredError):
        with_retry(
            lambda _model_id: (_ for _ in ()).throw(error),
            state=ModelRequestRecoveryState(current_model="primary"),
            repo_root=tmp_path,
            sleep_fn=sleeps.append,
            max_retry_after_seconds=60,
        )

    events = [json.loads(line) for line in trace_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    exhausted = [event for event in events if event["event"] == "model_retry_exhausted"][-1]
    assert sleeps == []
    assert exhausted["reason"] == "retry_after_exceeds_limit"


def test_finished_trace_does_not_report_unattempted_fallback_as_model(tmp_path):
    state = ModelRequestRecoveryState(
        current_model="primary",
        fallback_model_id="fallback",
        consecutive_529=2,
    )
    error = StructuredError("busy", status_code=529, headers={"Retry-After": "120"})

    with pytest.raises(StructuredError):
        with_retry(
            lambda _model_id: (_ for _ in ()).throw(error),
            state=state,
            repo_root=tmp_path,
            sleep_fn=lambda _delay: None,
        )

    events = [json.loads(line) for line in trace_path(tmp_path).read_text(encoding="utf-8").splitlines()]
    finished = [event for event in events if event["event"] == "model_request_finished"][-1]
    assert state.current_model == "fallback"
    assert finished["model"] == "primary"
    assert finished["selected_model"] == "fallback"


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
    assert response.metrics.model_calls == 2
    assert response.metrics.model_attempts == 2
    assert response.metrics.retries == 0


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
