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
from osc_agent.harness.background import (
    collect_background_results,
    should_run_background,
    start_background_task,
)
from osc_agent.harness.compact import COMPACT_TOOL, apply_compaction, compact_history, reactive_compact
from osc_agent.harness.cron import CRON_TOOLS, cancel_schedule, collect_cron_notifications, list_schedules, schedule_check
from osc_agent.harness.hooks import HookContext, HookRegistry, default_hooks, elapsed_ms
from osc_agent.harness.prompt import get_system_prompt, update_context
from osc_agent.harness.protocols import (
    PROTOCOL_TOOLS,
    request_plan_review,
    request_shutdown,
    request_write_approval,
    review_plan,
)
from osc_agent.harness.recovery import (
    CONTINUATION_PROMPT,
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    MAX_CONTINUATIONS,
    RecoveryState,
    is_prompt_too_long_error,
    with_retry,
)
from osc_agent.harness.subagent import SUBAGENT_TOOL, spawn_subagent
from osc_agent.harness.tasks import CONTRIBUTION_TASK_TOOLS, claim_task, complete_task, create_task, get_task, list_tasks
from osc_agent.harness.teams import TEAM_TOOLS, check_inbox, collect_team_notifications, send_message, spawn_teammate
from osc_agent.harness.todo import TODO_WRITE_TOOL, todo_write
from osc_agent.harness.trace import append_trace
from osc_agent.harness.worktree import WORKTREE_TOOLS, create_worktree, keep_worktree, remove_worktree
from osc_agent.skills.registry import LOAD_SKILL_TOOL, load_skill
from osc_agent.tools.files import FILE_TOOLS, edit_file, glob_files, read_file, write_file
from osc_agent.tools.git import GIT_TOOLS, git_diff, git_log, git_status
from osc_agent.tools.repo import REPO_TOOLS, inspect_repo
from osc_agent.tools.shell import BASH_TOOL, run_bash

TOOLS = [
    BASH_TOOL,
    *FILE_TOOLS,
    *GIT_TOOLS,
    *REPO_TOOLS,
    TODO_WRITE_TOOL,
    SUBAGENT_TOOL,
    LOAD_SKILL_TOOL,
    COMPACT_TOOL,
    *CRON_TOOLS,
    *TEAM_TOOLS,
    *PROTOCOL_TOOLS,
    *WORKTREE_TOOLS,
    *CONTRIBUTION_TASK_TOOLS,
]


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
    def subagent_handler(description: str, role: str) -> str:
        if client is None or settings is None:
            return "Error: subagent tool requires an agent client and settings"
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
        "bash": lambda command, run_in_background=False: run_bash(command, repo_root=repo_root, enforce_permissions=False),
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
        "subagent": subagent_handler,
        "load_skill": lambda name: load_skill(name),
        "compact": compact_handler,
        "spawn_teammate": lambda name, role, prompt, allow_write=False: spawn_teammate(
            name=name,
            role=role,
            prompt=prompt,
            allow_write=allow_write,
            repo_root=repo_root,
            client=client,
            settings=settings,
        ),
        "send_message": lambda to_agent, content, message_type="message", metadata=None: send_message(
            repo_root=repo_root,
            to_agent=to_agent,
            content=content,
            message_type=message_type,
            metadata=metadata,
        ),
        "check_inbox": lambda: check_inbox(repo_root=repo_root),
        "request_shutdown": lambda target, reason="": request_shutdown(
            repo_root=repo_root,
            target=target,
            reason=reason,
        ),
        "request_plan_review": lambda sender, plan: request_plan_review(
            repo_root=repo_root,
            sender=sender,
            plan=plan,
        ),
        "review_plan": lambda request_id, approve, feedback="": review_plan(
            repo_root=repo_root,
            request_id=request_id,
            approve=approve,
            feedback=feedback,
        ),
        "request_write_approval": lambda sender, path, reason: request_write_approval(
            repo_root=repo_root,
            sender=sender,
            path=path,
            reason=reason,
        ),
        "create_worktree": lambda name, task_id="": create_worktree(repo_root=repo_root, name=name, task_id=task_id),
        "keep_worktree": lambda name: keep_worktree(repo_root=repo_root, name=name),
        "remove_worktree": lambda name, discard_changes=False: remove_worktree(
            repo_root=repo_root,
            name=name,
            discard_changes=discard_changes,
        ),
        "schedule_check": lambda cron, prompt, enabled=True: schedule_check(
            repo_root=repo_root,
            cron=cron,
            prompt=prompt,
            enabled=enabled,
        ),
        "list_schedules": lambda: list_schedules(repo_root=repo_root),
        "cancel_schedule": lambda schedule_id: cancel_schedule(repo_root=repo_root, schedule_id=schedule_id),
        "create_task": lambda subject, description="", blockedBy=None, files=None, evidence=None, worktree=None: create_task(
            repo_root=repo_root,
            subject=subject,
            description=description,
            blockedBy=blockedBy,
            files=files,
            evidence=evidence,
            worktree=worktree,
        ),
        "list_tasks": lambda: list_tasks(repo_root=repo_root),
        "get_task": lambda task_id: get_task(repo_root=repo_root, task_id=task_id),
        "claim_task": lambda task_id, owner="agent": claim_task(repo_root=repo_root, task_id=task_id, owner=owner),
        "complete_task": lambda task_id, evidence=None: complete_task(
            repo_root=repo_root,
            task_id=task_id,
            evidence=evidence,
        ),
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

    recovery_state = RecoveryState(
        current_model=settings.model_id,
        fallback_model_id=settings.fallback_model_id,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    while True:
        notifications = (
            collect_background_results()
            + collect_cron_notifications(repo_root)
            + collect_team_notifications(repo_root)
        )
        if notifications:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": notification} for notification in notifications],
                }
            )

        messages[:] = apply_compaction(messages, repo_root=repo_root)
        prompt_context = update_context(
            repo_root=repo_root,
            messages=messages,
            enabled_tools=[tool["name"] for tool in TOOLS],
        )
        system_prompt = get_system_prompt(prompt_context)
        try:
            response = with_retry(
                lambda model_id: client.messages.create(
                    model=model_id,
                    system=system_prompt,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=recovery_state.max_tokens,
                ),
                state=recovery_state,
                repo_root=repo_root,
            )
        except Exception as exc:
            if not recovery_state.attempted_reactive_compact and is_prompt_too_long_error(exc):
                messages[:] = reactive_compact(messages, repo_root=repo_root)
                recovery_state.attempted_reactive_compact = True
                append_trace(repo_root, "prompt_too_long_recovery", {"action": "reactive_compact"})
                continue
            raise

        if response.stop_reason == "max_tokens":
            if not recovery_state.has_escalated_tokens:
                recovery_state.max_tokens = ESCALATED_MAX_TOKENS
                recovery_state.has_escalated_tokens = True
                append_trace(repo_root, "max_tokens_recovery", {"action": "escalate", "max_tokens": ESCALATED_MAX_TOKENS})
                continue

            messages.append({"role": "assistant", "content": response.content})
            if recovery_state.continuation_count < MAX_CONTINUATIONS:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                recovery_state.continuation_count += 1
                append_trace(
                    repo_root,
                    "max_tokens_recovery",
                    {"action": "continue", "count": recovery_state.continuation_count},
                )
                continue
            append_trace(repo_root, "max_tokens_recovery", {"action": "stop", "count": recovery_state.continuation_count})
            return response

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
                elif should_run_background(tool_name, tool_args):
                    # 后台任务只返回占位结果；真实输出由下一轮 task_notification 独立注入。
                    foreground_args = dict(tool_args)
                    foreground_args.pop("run_in_background", None)
                    task_id = start_background_task(
                        command=str(tool_args.get("command", "")),
                        repo_root=repo_root,
                        runner=lambda handler=handler, args=foreground_args: handler(**args),
                    )
                    tool_output = (
                        f"[Background task {task_id} started] "
                        "Result will be available in a later task_notification."
                    )
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
