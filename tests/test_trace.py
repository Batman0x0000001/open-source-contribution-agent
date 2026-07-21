from __future__ import annotations

import json

from osc_agent.harness.trace import (
    MAX_ARGUMENT_COLLECTION_ITEMS,
    MAX_ARGUMENT_DEPTH,
    MAX_ARGUMENT_STRING_CHARS,
    append_trace,
    sanitize_trace_text,
    sanitize_tool_arguments,
    trace_path,
)


def test_append_trace_appends_jsonl_without_overwriting(tmp_path):
    append_trace(tmp_path, "first", {"value": 1})
    append_trace(tmp_path, "second", {"value": 2})

    lines = trace_path(tmp_path).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "first"
    assert json.loads(lines[1])["event"] == "second"


def test_tool_arguments_are_redacted_before_tracing():
    sanitized = sanitize_tool_arguments(
        "write_file",
        {"path": "agent.py", "content": "secret source", "api_key": "abc"},
    )

    assert sanitized["content"]["chars"] == 13
    assert "secret source" not in json.dumps(sanitized)
    assert sanitized["api_key"] == "[REDACTED]"


def test_trace_text_redacts_authorization_and_limits_length():
    sanitized = sanitize_trace_text(
        "Authorization: Bearer top-secret api key=another-secret sk-ant-abcdefghijk "
        + "x" * 2_000,
        limit=200,
    )

    assert "top-secret" not in sanitized
    assert "another-secret" not in sanitized
    assert "sk-ant-abcdefghijk" not in sanitized
    assert "[REDACTED]" in sanitized
    assert len(sanitized) <= 200


def test_tool_arguments_are_bounded_by_length_depth_and_item_count():
    nested = {"secret": "value"}
    for _ in range(MAX_ARGUMENT_DEPTH + 1):
        nested = {"level": nested}
    arguments = {
        "long": "x" * (MAX_ARGUMENT_STRING_CHARS + 100),
        "nested": nested,
        **{f"key-{index}": index for index in range(MAX_ARGUMENT_COLLECTION_ITEMS + 10)},
    }

    sanitized = sanitize_tool_arguments("bash", arguments)
    encoded = json.dumps(sanitized, ensure_ascii=False)

    assert len(sanitized["long"]) == MAX_ARGUMENT_STRING_CHARS
    assert "maximum argument depth reached" in encoded
    assert sanitized["__truncated_items__"] == 12
