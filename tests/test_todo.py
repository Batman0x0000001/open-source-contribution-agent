from __future__ import annotations

import json

import pytest

from osc_agent.agent_loop import TOOLS, build_tool_handlers
from osc_agent.harness.todo import current_todos, todo_write
from osc_agent.harness.trace import trace_path


def test_todo_write_updates_current_plan_and_trace(tmp_path):
    output = todo_write(
        [
            {"content": "Read contribution guide", "status": "completed", "evidence": "README.md"},
            {"content": "Implement focused fix", "status": "in_progress"},
            {"content": "Run tests", "status": "pending"},
        ],
        repo_root=tmp_path,
    )

    assert "Implement focused fix" in output
    assert current_todos(tmp_path)[1] == {"content": "Implement focused fix", "status": "in_progress"}

    line = trace_path(tmp_path).read_text(encoding="utf-8").splitlines()[-1]
    event = json.loads(line)
    assert event["event"] == "todo_write"
    assert event["summary"] == {
        "total": 3,
        "pending": 1,
        "in_progress": 1,
        "completed": 1,
    }


def test_todo_write_rejects_multiple_in_progress():
    with pytest.raises(ValueError, match="only one todo"):
        todo_write(
            [
                {"content": "First", "status": "in_progress"},
                {"content": "Second", "status": "in_progress"},
            ]
        )


def test_todo_write_accepts_json_array_string():
    output = todo_write('[{"content": "Read docs", "status": "pending"}]')

    assert "Read docs" in output


def test_todos_are_isolated_by_repository(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    todo_write([{"content": "First task", "status": "pending"}], repo_root=first)
    todo_write([{"content": "Second task", "status": "pending"}], repo_root=second)

    assert current_todos(first)[0]["content"] == "First task"
    assert current_todos(second)[0]["content"] == "Second task"


def test_todo_write_accepts_python_list_repr_without_eval():
    output = todo_write("[{'content': 'Read guide', 'status': 'completed'}]")

    assert "Read guide" in output


def test_todo_write_rejects_non_list_string():
    with pytest.raises(ValueError, match="parse to a list"):
        todo_write('{"content": "not a list", "status": "pending"}')


def test_todo_write_rejects_invalid_status():
    with pytest.raises(ValueError, match="status"):
        todo_write([{"content": "Ship it", "status": "blocked"}])


def test_agent_loop_exposes_todo_write_handler(tmp_path):
    tool_names = {tool["name"] for tool in TOOLS}
    handlers = build_tool_handlers(tmp_path)

    assert "todo_write" in tool_names
    assert "todo_write" in handlers
    assert "Plan task" in handlers["todo_write"](
        [{"content": "Plan task", "status": "in_progress"}]
    )
