"""
思考 / 回复
  ↓
请求工具
  ↓
检查权限
  ↓
执行命令
  ↓
把工具结果交回模型
  ↓
直到模型不再请求工具
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TextIO

from osc_agent.config import Settings
from osc_agent.harness.compact import COMPACT_TOOL, apply_compaction, compact_history, reactive_compact
from osc_agent.harness.hooks import HookContext, HookRegistry, default_hooks, elapsed_ms
from osc_agent.harness.prompt import get_system_prompt, update_context
from osc_agent.harness.todo import TODO_WRITE_TOOL, todo_write
from osc_agent.skills.registry import LOAD_SKILL_TOOL, load_skill
from osc_agent.tools.files import FILE_TOOLS, edit_file, glob_files, read_file, write_file
from osc_agent.tools.git import GIT_TOOLS, git_diff, git_log, git_status
from osc_agent.tools.repo import REPO_TOOLS, inspect_repo
from osc_agent.tools.shell import BASH_TOOL, run_bash
from osc_agent.tools.task import TASK_TOOL, spawn_subagent

TOOLS = [BASH_TOOL, *FILE_TOOLS, *GIT_TOOLS, *REPO_TOOLS, TODO_WRITE_TOOL, TASK_TOOL, LOAD_SKILL_TOOL, COMPACT_TOOL]


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    """兼容 Anthropic SDK 对象和测试里的 dict block，降低 mock 成本。"""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _tool_input(block: Any) -> dict[str, Any]:
    """提取 tool_use 输入参数，缺失时返回空 dict 以便主循环稳定处理。"""
    value = _block_attr(block, "input", {})
    return value if isinstance(value, dict) else {}


def build_tool_handlers(
    repo_root: Path,
    *,
    client: Any | None = None,
    settings: Settings | None = None,
    confirm: Callable[[str], bool] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按 repo_root 绑定工具函数，主循环只负责按名称分发。"""
    def task_handler(description: str, role: str) -> str:
        if client is None or settings is None:
            return "Error: task tool requires an agent client and settings"
        return spawn_subagent(
            description,
            role,
            client=client,
            settings=settings,
            repo_root=repo_root,
            confirm=confirm,
        )

    def compact_handler(reason: str = "manual") -> str:
        if messages is None:
            return "Error: compact tool requires active messages"
        messages[:] = compact_history(messages, repo_root=repo_root, reason=reason or "manual")
        return "[Compacted. History summarized.]"

    return {
        "bash": lambda command: run_bash(command, repo_root=repo_root, enforce_permissions=False),
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
            enforce_permissions=False,
        ),
        "edit_file": lambda path, old_text, new_text: edit_file(
            repo_root=repo_root,
            path=path,
            old_text=old_text,
            new_text=new_text,
            enforce_permissions=False,
        ),
        "glob": lambda pattern: glob_files(repo_root=repo_root, pattern=pattern),
        "git_status": lambda: git_status(repo_root=repo_root),
        "git_diff": lambda: git_diff(repo_root=repo_root),
        "git_log": lambda limit=5: git_log(repo_root=repo_root, limit=limit),
        "inspect_repo": lambda: inspect_repo(repo_root=repo_root),
        "todo_write": lambda todos: todo_write(todos, repo_root=repo_root),
        "task": task_handler,
        "load_skill": lambda name: load_skill(name),
        "compact": compact_handler,
    }


def agent_loop(
    messages: list[dict[str, Any]],
    *,
    client: Any,
    settings: Settings,
    repo_root: Path,
    output: TextIO | None = None,
    tool_handlers: dict[str, Any] | None = None,
    hooks: HookRegistry | None = None,
    confirm: Callable[[str], bool] | None = None,
) -> Any:
    """执行 Anthropic 风格 agent loop，直到模型不再请求工具。"""
    handlers = tool_handlers or build_tool_handlers(
        repo_root,
        client=client,
        settings=settings,
        confirm=confirm,
        messages=messages,
    )
    hook_registry = default_hooks()
    if hooks is not None:
        # 自定义 hook 只能追加，不能替换默认权限检查。
        hook_registry.extend(hooks)
    hook_context = HookContext(repo_root=repo_root, confirm=confirm)

    reactive_retries = 0

    while True:
        messages[:] = apply_compaction(messages, repo_root=repo_root)
        prompt_context = update_context(
            repo_root=repo_root,
            messages=messages,
            enabled_tools=[tool["name"] for tool in TOOLS],
        )
        system_prompt = get_system_prompt(prompt_context)
        try:
            response = client.messages.create(
                model=settings.model_id,
                system=system_prompt,
                messages=messages,
                tools=TOOLS,
                max_tokens=8000,
            )
        except Exception as exc:
            if reactive_retries < 1 and _is_prompt_too_long(exc):
                messages[:] = reactive_compact(messages, repo_root=repo_root)
                reactive_retries += 1
                continue
            raise
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            hook_registry.run("Stop", hook_context, {"stop_reason": response.stop_reason})
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
                pre_results = hook_registry.run(
                    "PreToolUse",
                    hook_context,
                    {"tool_name": tool_name, "tool_args": tool_args},
                )
                blocked = next((result for result in pre_results if not result.allowed), None)
                started = perf_counter()
                if blocked is not None:
                    tool_output = blocked.content or "Permission denied"
                else:
                    try:
                        tool_output = handler(**tool_args)
                    except (TypeError, ValueError) as exc:
                        tool_output = f"Error: invalid arguments for {tool_name}: {exc}"
                hook_registry.run(
                    "PostToolUse",
                    hook_context,
                    {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "output": tool_output,
                        "latency_ms": elapsed_ms(started),
                    },
                )
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


def _is_prompt_too_long(exc: Exception) -> bool:
    text = str(exc).lower()
    return "prompt_too_long" in text or "prompt too long" in text or "context length" in text
