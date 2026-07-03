"""
模型调用 todo_write
    ↓
解析 todos
    ↓
校验并标准化 todos
    ↓
更新全局 CURRENT_TODOS
    ↓
可选写入 trace
    ↓
渲染成文本返回给模型
"""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from typing import Any

from osc_agent.harness.trace import append_trace

TODO_STATUSES = {"pending", "in_progress", "completed"}

TODO_WRITE_TOOL = {
    "name": "todo_write",
    "description": "Create and update the current contribution task plan.",
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                        "evidence": {"type": "string"},
                    },
                    "required": ["content", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["todos"],
        "additionalProperties": False,
    },
}

CURRENT_TODOS: list[dict[str, str]] = []


def todo_write(todos: list[dict[str, Any]] | str, *, repo_root: Any | None = None) -> str:
    """更新当前 TODO 计划，并确保同一时间只有一个任务处于进行中。"""
    normalized = _normalize_todos(_parse_todos(todos))

    global CURRENT_TODOS
    CURRENT_TODOS = normalized

    if repo_root is not None:
        append_trace(
            repo_root,
            "todo_write",
            {
                "todos": deepcopy(CURRENT_TODOS),
                "summary": summarize_todos(CURRENT_TODOS),
            },
        )

    return _render_todos(CURRENT_TODOS)


def current_todos() -> list[dict[str, str]]:
    """返回 TODO 快照，避免调用方直接修改模块内全局状态。"""
    return deepcopy(CURRENT_TODOS)


def summarize_todos(todos: list[dict[str, str]] | None = None) -> dict[str, int]:
    source = CURRENT_TODOS if todos is None else todos
    return {
        "total": len(source),
        "pending": sum(1 for todo in source if todo["status"] == "pending"),
        "in_progress": sum(1 for todo in source if todo["status"] == "in_progress"),
        "completed": sum(1 for todo in source if todo["status"] == "completed"),
    }


def _parse_todos(todos: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(todos, list):
        return todos
    if not isinstance(todos, str):
        raise ValueError("todos must be a list or string")

    text = todos.strip()
    if not text:
        raise ValueError("todos string must not be empty")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            # 支持 Python list repr，但只用 literal_eval 解析字面量，禁止 eval 执行代码。
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("todos string must be a JSON array or Python list literal") from exc

    if not isinstance(parsed, list):
        raise ValueError("todos must parse to a list")
    return parsed


def _normalize_todos(todos: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(todos, list):
        raise ValueError("todos must be a list")

    normalized: list[dict[str, str]] = []
    in_progress_count = 0
    for index, todo in enumerate(todos, start=1):
        if not isinstance(todo, dict):
            raise ValueError(f"todo #{index} must be an object")

        content = todo.get("content")
        status = todo.get("status")
        evidence = todo.get("evidence")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"todo #{index} content must be a non-empty string")
        if status not in TODO_STATUSES:
            raise ValueError(f"todo #{index} status must be pending, in_progress, or completed")
        if evidence is not None and not isinstance(evidence, str):
            raise ValueError(f"todo #{index} evidence must be a string")

        if status == "in_progress":
            in_progress_count += 1

        # 只保留 harness 认可的字段，防止模型把临时数据混进当前计划。
        normalized_todo = {"content": content.strip(), "status": status}
        if isinstance(evidence, str) and evidence.strip():
            normalized_todo["evidence"] = evidence.strip()
        normalized.append(normalized_todo)

    if in_progress_count > 1:
        raise ValueError("only one todo can be in_progress")

    return normalized


def _render_todos(todos: list[dict[str, str]]) -> str:
    lines = ["## Current Tasks"]
    if not todos:
        lines.append("(no tasks)")
        return "\n".join(lines)

    icon_by_status = {
        "pending": " ",
        "in_progress": ">",
        "completed": "x",
    }
    for todo in todos:
        line = f"[{icon_by_status[todo['status']]}] {todo['content']}"
        #等价于evidence = todo.get("evidence") \n if evidence:
        if evidence := todo.get("evidence"):
            line = f"{line} ({evidence})"
        lines.append(line)
    return "\n".join(lines)
