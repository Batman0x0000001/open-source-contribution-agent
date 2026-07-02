"""
思考 / 回复
  ↓
请求 bash 工具
  ↓
执行命令
  ↓
把工具结果交回模型
  ↓
直到模型不再请求工具
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from osc_agent.config import Settings
from osc_agent.tools.shell import BASH_TOOL, run_bash

SYSTEM_TEMPLATE = (
    "You are a coding agent working inside this local repository: {repo_root}. "
    "Use bash to inspect and solve the user's contribution task. "
    "Act step by step and stop when you can report the result."
)


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    """兼容 Anthropic SDK 对象和测试里的 dict block，降低 mock 成本。"""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _tool_input(block: Any) -> dict[str, Any]:
    """提取 tool_use 输入参数，缺失时返回空 dict 以便主循环稳定处理。"""
    value = _block_attr(block, "input", {})
    return value if isinstance(value, dict) else {}


def agent_loop(
    messages: list[dict[str, Any]],
    *,
    client: Any,
    settings: Settings,
    repo_root: Path,
    output: TextIO | None = None,
    tool_runner: Callable[[str], str] | None = None,
) -> Any:
    """执行 Anthropic 风格 agent loop，直到模型不再请求工具。"""
    system_prompt = SYSTEM_TEMPLATE.format(repo_root=repo_root)
    run_tool = tool_runner or (lambda command: run_bash(command, repo_root=repo_root))

    while True:
        response = client.messages.create(
            model=settings.model_id,
            system=system_prompt,
            messages=messages,
            tools=[BASH_TOOL],
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return response

        results: list[dict[str, str]] = []
        for block in response.content:
            if _block_attr(block, "type") != "tool_use":
                continue

            tool_name = _block_attr(block, "name")
            tool_use_id = _block_attr(block, "id")
            command = str(_tool_input(block).get("command", ""))

            if tool_name != "bash":
                tool_output = f"Error: unknown tool {tool_name}"
            else:
                if output is not None:
                    print(f"$ {command}", file=output)
                tool_output = run_tool(command)
                if output is not None:
                    print(tool_output[:200], file=output)

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_output,
                }
            )

        messages.append({"role": "user", "content": results})
