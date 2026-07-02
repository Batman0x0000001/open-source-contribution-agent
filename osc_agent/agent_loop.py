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

from pathlib import Path
from typing import Any, TextIO

from osc_agent.config import Settings
from osc_agent.tools.files import FILE_TOOLS, edit_file, glob_files, read_file, write_file
from osc_agent.tools.git import GIT_TOOLS, git_diff, git_log, git_status
from osc_agent.tools.repo import REPO_TOOLS, inspect_repo
from osc_agent.tools.shell import BASH_TOOL, run_bash

SYSTEM_TEMPLATE = (
    "You are a coding agent working inside this local repository: {repo_root}. "
    "Use the repo, file, git, and bash tools to inspect and solve the user's contribution task. "
    "Act step by step and stop when you can report the result."
)

TOOLS = [BASH_TOOL, *FILE_TOOLS, *GIT_TOOLS, *REPO_TOOLS]


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    """兼容 Anthropic SDK 对象和测试里的 dict block，降低 mock 成本。"""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _tool_input(block: Any) -> dict[str, Any]:
    """提取 tool_use 输入参数，缺失时返回空 dict 以便主循环稳定处理。"""
    value = _block_attr(block, "input", {})
    return value if isinstance(value, dict) else {}


def build_tool_handlers(repo_root: Path) -> dict[str, Any]:
    """按 repo_root 绑定工具函数，主循环只负责按名称分发。"""
    return {
        "bash": lambda command: run_bash(command, repo_root=repo_root),
        "read_file": lambda path, limit=20_000, offset=0: read_file(
            repo_root=repo_root,
            path=path,
            limit=limit,
            offset=offset,
        ),
        "write_file": lambda path, content: write_file(
            repo_root=repo_root,
            path=path,
            content=content,
        ),
        "edit_file": lambda path, old_text, new_text: edit_file(
            repo_root=repo_root,
            path=path,
            old_text=old_text,
            new_text=new_text,
        ),
        "glob": lambda pattern: glob_files(repo_root=repo_root, pattern=pattern),
        "git_status": lambda: git_status(repo_root=repo_root),
        "git_diff": lambda: git_diff(repo_root=repo_root),
        "git_log": lambda limit=5: git_log(repo_root=repo_root, limit=limit),
        "inspect_repo": lambda: inspect_repo(repo_root=repo_root),
    }


def agent_loop(
    messages: list[dict[str, Any]],
    *,
    client: Any,
    settings: Settings,
    repo_root: Path,
    output: TextIO | None = None,
    tool_handlers: dict[str, Any] | None = None,
) -> Any:
    """执行 Anthropic 风格 agent loop，直到模型不再请求工具。"""
    system_prompt = SYSTEM_TEMPLATE.format(repo_root=repo_root)
    handlers = tool_handlers or build_tool_handlers(repo_root)

    while True:
        response = client.messages.create(
            model=settings.model_id,
            system=system_prompt,
            messages=messages,
            tools=TOOLS,
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
            tool_args = _tool_input(block)

            handler = handlers.get(tool_name)
            if handler is None:
                tool_output = f"Error: unknown tool {tool_name}"
            else:
                if output is not None:
                    print(f"{tool_name}: {tool_args}", file=output)
                try:
                    tool_output = handler(**tool_args)
                except TypeError as exc:
                    tool_output = f"Error: invalid arguments for {tool_name}: {exc}"
                if output is not None:
                    print(str(tool_output)[:200], file=output)

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_output,
                }
            )

        #Anthropic 把工具看成：用户帮模型完成了一件事，然后把结果告诉模型
        messages.append({"role": "user", "content": results})
