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
from time import monotonic, perf_counter
from typing import Any, Callable, TextIO

from osc_agent.config import Settings
from osc_agent.harness.background import (
    collect_background_results,
    should_run_background,
    start_background_task,
)
from osc_agent.harness.compact import COMPACT_TOOL, apply_compaction, compact_history, reactive_compact
from osc_agent.harness.contracts import (
    AgentRunResult,
    RunMetrics,
    RunStatus,
    action_fingerprint,
    normalize_tool_result,
)
from osc_agent.harness.cron import CRON_TOOLS, cancel_schedule, collect_cron_notifications, list_schedules, schedule_check
from osc_agent.harness.hooks import HookContext, HookRegistry, default_hooks, elapsed_ms
from osc_agent.harness.mcp import CONNECT_MCP_TOOL, assemble_tool_handlers, assemble_tool_pool, connect_mcp
from osc_agent.harness.prompt import assemble_system_prompt, update_context
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
from osc_agent.harness.runtime_state import record_tool_observation
from osc_agent.harness.subagent import SUBAGENT_TOOL, spawn_subagent
from osc_agent.harness.tasks import CONTRIBUTION_TASK_TOOLS, claim_task, complete_task, create_task, get_task, list_tasks
from osc_agent.harness.teams import TEAM_TOOLS, check_inbox, collect_team_notifications, send_message, spawn_teammate
from osc_agent.harness.todo import TODO_WRITE_TOOL, todo_write
from osc_agent.harness.trace import append_trace
from osc_agent.harness.worktree import WORKTREE_TOOLS, create_worktree, keep_worktree, remove_worktree
from osc_agent.skills.registry import LOAD_SKILL_TOOL, load_skill
from osc_agent.tools.files import FILE_TOOLS, edit_file, glob_files, read_file, write_file
from osc_agent.tools.git import GIT_TOOLS, git_diff, git_log, git_status
from osc_agent.tools.pr import PR_TOOLS, draft_pr
from osc_agent.tools.repo import REPO_TOOLS, inspect_repo
from osc_agent.tools.shell import BASH_TOOL, run_bash

TOOLS = [
    BASH_TOOL,
    *FILE_TOOLS,
    *GIT_TOOLS,
    *PR_TOOLS,
    *REPO_TOOLS,
    TODO_WRITE_TOOL,
    SUBAGENT_TOOL,
    LOAD_SKILL_TOOL,
    COMPACT_TOOL,
    CONNECT_MCP_TOOL,
    *CRON_TOOLS,
    *TEAM_TOOLS,
    *PROTOCOL_TOOLS,
    *WORKTREE_TOOLS,
    *CONTRIBUTION_TASK_TOOLS,
]

SIDE_EFFECT_TOOLS = {
    "bash",
    "write_file",
    "edit_file",
    "todo_write",
    "connect_mcp",
    "spawn_teammate",
    "send_message",
    "request_shutdown",
    "request_plan_review",
    "review_plan",
    "request_write_approval",
    "create_worktree",
    "keep_worktree",
    "remove_worktree",
    "schedule_check",
    "cancel_schedule",
    "create_task",
    "claim_task",
    "complete_task",
}


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
        "draft_pr": lambda: draft_pr(repo_root=repo_root),
        "inspect_repo": lambda: inspect_repo(repo_root=repo_root),
        "todo_write": lambda todos: todo_write(todos, repo_root=repo_root),
        "subagent": subagent_handler,
        "load_skill": lambda name: load_skill(name),
        "compact": compact_handler,
        "connect_mcp": lambda server_name: connect_mcp(server_name),
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
) -> AgentRunResult:
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
    metrics = RunMetrics()
    started_at = monotonic()
    previous_fingerprint: str | None = None
    repeated_actions = 0
    consecutive_failures = 0
    seen_fingerprints: set[str] = set()
    no_progress_calls = 0
    budget_overrides: set[str] = set()

    while True:
        metrics.elapsed_ms = int((monotonic() - started_at) * 1000)
        budget_reason = _budget_reason(metrics, settings, confirm=confirm, overrides=budget_overrides)
        if budget_reason:
            return _finish_run(repo_root, RunStatus.FAILED_BUDGET, budget_reason, metrics, None)

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
        active_tools = assemble_tool_pool(TOOLS)
        active_handlers = assemble_tool_handlers(handlers)
        prompt_context = update_context(
            repo_root=repo_root,
            messages=messages,
            enabled_tools=[tool["name"] for tool in active_tools],
        )
        # MCP 工具池会在 agent_loop 运行中动态变化，因此这里每轮直接重组 system prompt。
        system_prompt = assemble_system_prompt(prompt_context)
        try:
            metrics.model_calls += 1
            response = with_retry(
                lambda model_id: client.messages.create(
                    model=model_id,
                    system=system_prompt,
                    messages=messages,
                    tools=active_tools,
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

        metrics.retries = recovery_state.retry_count
        input_tokens, output_tokens = _response_usage(response)
        metrics.input_tokens += input_tokens
        metrics.output_tokens += output_tokens
        metrics.elapsed_ms = int((monotonic() - started_at) * 1000)
        budget_reason = _budget_reason(
            metrics,
            settings,
            check_rounds=False,
            confirm=confirm,
            overrides=budget_overrides,
        )
        if budget_reason:
            messages.append({"role": "assistant", "content": response.content})
            return _finish_run(repo_root, RunStatus.FAILED_BUDGET, budget_reason, metrics, response)

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
            return _finish_run(
                repo_root,
                RunStatus.FAILED_BUDGET,
                "model output continuation budget exhausted",
                metrics,
                response,
            )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            hook_registry.run("Stop", hook_context, {"stop_reason": response.stop_reason})
            return _finish_run(repo_root, RunStatus.SUCCESS, "model completed without tool requests", metrics, response)

        results: list[dict[str, str]] = []
        for block in response.content:
            if _block_attr(block, "type") != "tool_use":
                continue

            tool_name = _block_attr(block, "name")
            tool_use_id = _block_attr(block, "id")
            tool_args = _tool_input(block)
            fingerprint = action_fingerprint(str(tool_name), tool_args)
            if fingerprint == previous_fingerprint:
                repeated_actions += 1
            else:
                previous_fingerprint = fingerprint
                repeated_actions = 1
            if repeated_actions >= settings.repeat_action_limit:
                return _finish_run(
                    repo_root,
                    RunStatus.BLOCKED_NEEDS_USER,
                    f"tool action repeated {repeated_actions} times: {tool_name}",
                    metrics,
                    response,
                )

            handler = active_handlers.get(tool_name)
            started = perf_counter()
            if handler is None:
                raw_output = f"Error: unknown tool {tool_name}"
            else:
                if output is not None:
                    print(f"{tool_name}: {tool_args}", file=output)
                pre_results = hook_registry.run(
                    "PreToolUse",
                    hook_context,
                    {"tool_name": tool_name, "tool_args": tool_args},
                )
                blocked = next((result for result in pre_results if not result.allowed), None)
                if blocked is not None:
                    raw_output = blocked.content or "Permission denied"
                elif should_run_background(tool_name, tool_args):
                    # 后台任务只返回占位结果；真实输出由下一轮 task_notification 独立注入。
                    foreground_args = dict(tool_args)
                    foreground_args.pop("run_in_background", None)
                    task_id = start_background_task(
                        command=str(tool_args.get("command", "")),
                        repo_root=repo_root,
                        runner=lambda handler=handler, args=foreground_args: handler(**args),
                    )
                    raw_output = (
                        f"[Background task {task_id} started] "
                        "Result will be available in a later task_notification."
                    )
                else:
                    try:
                        raw_output = handler(**tool_args)
                    except (TypeError, ValueError) as exc:
                        raw_output = f"Error: invalid arguments for {tool_name}: {exc}"
                    except Exception as exc:  # noqa: BLE001 - 工具失败必须转成可审计结果
                        raw_output = f"Error: tool {tool_name} failed: {exc}"

            tool_result = normalize_tool_result(
                raw_output,
                tool_name=str(tool_name),
                arguments=tool_args,
                call_id=str(tool_use_id) if tool_use_id is not None else None,
                latency_ms=elapsed_ms(started),
                side_effect=str(tool_name) in SIDE_EFFECT_TOOLS,
            )
            tool_output = tool_result.to_model_content()
            record_tool_observation(repo_root, str(tool_name), tool_args, tool_result)
            metrics.tool_calls += 1
            if tool_result.ok:
                consecutive_failures = 0
                if fingerprint in seen_fingerprints and not tool_result.side_effect:
                    no_progress_calls += 1
                else:
                    no_progress_calls = 0
                seen_fingerprints.add(fingerprint)
            else:
                metrics.tool_failures += 1
                consecutive_failures += 1
                no_progress_calls += 1
            hook_registry.run(
                "PostToolUse",
                hook_context,
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "output": tool_result.to_dict(),
                    "latency_ms": tool_result.latency_ms,
                },
            )
            if output is not None:
                print(str(tool_output)[:200], file=output)

            if consecutive_failures >= settings.consecutive_failure_limit:
                return _finish_run(
                    repo_root,
                    RunStatus.FAILED_TOOL,
                    f"{consecutive_failures} consecutive tool failures",
                    metrics,
                    response,
                )
            if no_progress_calls >= settings.no_progress_limit:
                return _finish_run(
                    repo_root,
                    RunStatus.BLOCKED_NEEDS_USER,
                    f"no new evidence or state change after {no_progress_calls} tool calls",
                    metrics,
                    response,
                )

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_output,
                }
            )

        #Anthropic 把工具看成：用户帮模型完成了一件事，然后把结果告诉模型
        messages.append({"role": "user", "content": results})


def _response_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def _budget_reason(
    metrics: RunMetrics,
    settings: Settings,
    *,
    check_rounds: bool = True,
    confirm: Callable[[str], bool] | None = None,
    overrides: set[str] | None = None,
) -> str | None:
    granted = overrides if overrides is not None else set()

    if check_rounds and "rounds" not in granted and metrics.model_calls >= settings.max_agent_rounds:
        if confirm is not None and confirm(f"已达到最大轮次限制 ({settings.max_agent_rounds})，是否继续执行？"):
            granted.add("rounds")
        else:
            return f"maximum model rounds reached ({settings.max_agent_rounds})"
    if "tokens" not in granted and metrics.total_tokens >= settings.max_total_tokens:
        if confirm is not None and confirm(f"已达到 token 预算限制 ({settings.max_total_tokens})，是否继续执行？"):
            granted.add("tokens")
        else:
            return f"maximum token budget reached ({settings.max_total_tokens})"
    if "deadline" not in granted and metrics.elapsed_ms >= settings.agent_deadline_seconds * 1000:
        if confirm is not None and confirm(f"已达到时间限制 ({settings.agent_deadline_seconds}s)，是否继续执行？"):
            granted.add("deadline")
        else:
            return f"agent deadline reached ({settings.agent_deadline_seconds}s)"
    return None


def _finish_run(
    repo_root: Path,
    status: RunStatus,
    reason: str,
    metrics: RunMetrics,
    response: Any | None,
) -> AgentRunResult:
    append_trace(
        repo_root,
        "agent_run_finished",
        {"status": status.value, "reason": reason, "metrics": metrics.to_dict()},
    )
    return AgentRunResult(status=status, response=response, reason=reason, metrics=metrics)
