from __future__ import annotations

import json

from osc_agent.harness.trace import append_trace, trace_path


def test_append_trace_appends_jsonl_without_overwriting(tmp_path):
    append_trace(tmp_path, "first", {"value": 1})
    append_trace(tmp_path, "second", {"value": 2})

    lines = trace_path(tmp_path).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "first"
    assert json.loads(lines[1])["event"] == "second"
