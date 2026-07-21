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

from dataclasses import dataclass
from pathlib import Path
import secrets
from typing import Any, Callable, TextIO

from osc_agent.config import Settings
from osc_agent.harness.background import (
    collect_background_results,
)
from osc_agent.harness.compact import COMPACT_TOOL, apply_compaction, compact_history, reactive_compact
from osc_agent.harness.contracts import (
    AgentRunResult,
    RunMetrics,
    RunStatus,
)
from osc_agent.harness.cron import CRON_TOOLS, cancel_schedule, collect_cron_notifications, list_schedules, schedule_check
from osc_agent.harness.hooks import HookContext, HookRegistry, default_hooks
from osc_agent.harness.loop_state import LoopExecutionState
from osc_agent.harness.mcp import CONNECT_MCP_TOOL, connect_mcp
from osc_agent.harness.prompt import assemble_system_prompt, update_context
from osc_agent.harness.capabilities import AgentCapabilityScope
from osc_agent.harness.protocols import (
    PROTOCOL_TOOLS,
    request_plan_review,
    request_shutdown,
    request_write_approval,
    review_plan,
)
from osc_agent.harness.recovery import (
    CONTINUATION_PROMPT,
    ESCALATED_MAX_TOKENS,
    MAX_CONTINUATIONS,
    classify_model_error,
    is_prompt_too_long_error,
    safe_model_error,
    with_retry,
)
from osc_agent.harness.subagent import SUBAGENT_TOOL, spawn_subagent
from osc_agent.harness.tasks import CONTRIBUTION_TASK_TOOLS, claim_task, complete_task, create_task, get_task, list_tasks
from osc_agent.harness.teams import TEAM_TOOLS, check_inbox, collect_team_notifications, send_message, spawn_teammate
from osc_agent.harness.todo import TODO_WRITE_TOOL, todo_write
from osc_agent.harness.tool_execution import block_attr, execute_tool_call, parse_tool_call
from osc_agent.harness.tool_registry import ToolRegistry
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


def build_tool_handlers(
    repo_root: Path,
    *,
    client: Any | None = None,
    settings: Settings | None = None,
    confirm: Callable[[str], bool] | None = None,
    messages: list[dict[str, Any]] | None = None,
    session_id: str = "default",
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
        "bash": lambda command, run_in_background=False: run_bash(command, repo_root=repo_root, enforce_risk_checks=False),
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
            enforce_risk_checks=False,
        ),
        "edit_file": lambda path, old_text, new_text: edit_file(
            repo_root=repo_root,
            path=path,
            old_text=old_text,
            new_text=new_text,
            enforce_risk_checks=False,
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
        "connect_mcp": lambda server_name: connect_mcp(server_name, session_id=session_id),
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


@dataclass(frozen=True)
class PreparedRound:
    tools: list[dict[str, Any]]
    handlers: dict[str, Any]
    system_prompt: str


@dataclass(frozen=True)
class ModelRequestOutcome:
    response: Any | None = None
    retry_round: bool = False
    failure_reason: str | None = None


def _first_user_task(messages: list[dict[str, Any]]) -> str:
    """保留最初的文本任务，避免后续工具结果覆盖运行目标。"""
    return next(
        (
            str(message.get("content"))
            for message in messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )


def _prepare_round(
    messages: list[dict[str, Any]],
    *,
    repo_root: Path,
    objective: str,
    current_instruction: str,
    session_id: str,
    capabilities: AgentCapabilityScope,
    tool_registry: ToolRegistry,
) -> PreparedRound:
    notifications = (
        collect_background_results(repo_root)
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

    # 保持调用者持有的 messages 列表对象不变，只替换其中的压缩结果。
    messages[:] = apply_compaction(messages, repo_root=repo_root)
    tools = capabilities.filter_tools(tool_registry.schemas(session_id=session_id))
    # Schema 控制模型可见能力；Handler 保持完整，确保越权调用仍由 Capability Hook 明确拒绝。
    handlers = tool_registry.handlers(session_id=session_id)
    prompt_context = update_context(
        repo_root=repo_root,
        objective=objective,
        current_instruction=current_instruction,
        enabled_tools=[tool["name"] for tool in tools],
        capabilities=capabilities,
        run_id=capabilities.run_id,
        session_id=session_id,
    )
    # MCP 工具可在运行中接入，因此每轮都必须重新构造工具池和 system prompt。
    return PreparedRound(
        tools=tools,
        handlers=handlers,
        system_prompt=assemble_system_prompt(prompt_context),
    )


def _request_model(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    prepared: PreparedRound,
    state: LoopExecutionState,
    repo_root: Path,
) -> ModelRequestOutcome:
    state.metrics.model_calls += 1
    try:
        response = with_retry(
            lambda model_id: client.messages.create(
                model=model_id,
                system=prepared.system_prompt,
                messages=messages,
                tools=prepared.tools,
                max_tokens=state.response_recovery.max_tokens,
            ),
            state=state.request_recovery,
            repo_root=repo_root,
        )
    except Exception as exc:
        if (
            not state.response_recovery.attempted_reactive_compact
            and is_prompt_too_long_error(exc)
        ):
            messages[:] = reactive_compact(messages, repo_root=repo_root)
            state.response_recovery.attempted_reactive_compact = True
            append_trace(repo_root, "prompt_too_long_recovery", {"action": "reactive_compact"})
            return ModelRequestOutcome(retry_round=True)
        safe_error = safe_model_error(exc)
        return ModelRequestOutcome(
            failure_reason=(
                f"model request failed: {classify_model_error(exc)}: {safe_error['error']}"
            )
        )
    finally:
        state.metrics.retries = state.request_recovery.retry_count
        state.metrics.model_attempts = state.request_recovery.total_attempts
        state.metrics.fallback_switches = state.request_recovery.fallback_switches
    return ModelRequestOutcome(response=response)


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
    capabilities: AgentCapabilityScope | None = None,
    session_id: str | None = None,
    objective: str | None = None,
) -> AgentRunResult:
    """执行 Anthropic 风格 agent loop，直到模型不再请求工具。"""
    active_session_id = session_id or f"session_{secrets.token_hex(8)}"
    handlers = (
        tool_handlers
        if tool_handlers is not None
        else build_tool_handlers(
            repo_root,
            client=client,
            settings=settings,
            confirm=confirm,
            messages=messages,
            session_id=active_session_id,
        )
    )
    tool_registry = ToolRegistry(
        TOOLS,
        handlers,
        side_effect_tools=SIDE_EFFECT_TOOLS,
        require_complete_handlers=tool_handlers is None,
    )
    hook_registry = default_hooks()
    if hooks is not None:
        # 自定义 hook 只能追加，不能替换默认安全检查。
        hook_registry.extend(hooks)
    capability_scope = capabilities or AgentCapabilityScope.unrestricted()
    current_instruction = _first_user_task(messages)
    top_level_objective = objective or current_instruction
    hook_context = HookContext(
        repo_root=repo_root,
        confirm=confirm,
        capabilities=capability_scope,
        session_id=active_session_id,
        run_id=capability_scope.run_id,
    )
    state = LoopExecutionState.from_settings(settings)

    def finish(status: RunStatus, reason: str, response: Any | None) -> AgentRunResult:
        if not state.stopped:
            hook_registry.run("Stop", hook_context, {"stop_reason": reason, "status": status.value})
            state.stopped = True
        return _finish_run(
            repo_root,
            status,
            reason,
            state.metrics,
            response,
            session_id=active_session_id,
            run_id=capability_scope.run_id,
        )

    while True:
        state.update_elapsed()
        budget_reason = _budget_reason(
            state.metrics,
            settings,
            confirm=confirm,
            overrides=state.budget_overrides,
        )
        if budget_reason:
            return finish(RunStatus.FAILED_BUDGET, budget_reason, None)

        prepared = _prepare_round(
            messages,
            repo_root=repo_root,
            objective=top_level_objective,
            current_instruction=current_instruction,
            session_id=active_session_id,
            capabilities=capability_scope,
            tool_registry=tool_registry,
        )
        request = _request_model(
            client=client,
            messages=messages,
            prepared=prepared,
            state=state,
            repo_root=repo_root,
        )
        if request.retry_round:
            continue
        if request.failure_reason is not None:
            return finish(
                RunStatus.FAILED_TOOL,
                request.failure_reason,
                None,
            )
        response = request.response
        if response is None:
            return finish(RunStatus.FAILED_TOOL, "model request returned no response", None)

        input_tokens, output_tokens = _response_usage(response)
        state.metrics.input_tokens += input_tokens
        state.metrics.output_tokens += output_tokens
        state.update_elapsed()
        budget_reason = _budget_reason(
            state.metrics,
            settings,
            check_rounds=False,
            confirm=confirm,
            overrides=state.budget_overrides,
        )
        if budget_reason:
            messages.append({"role": "assistant", "content": response.content})
            return finish(RunStatus.FAILED_BUDGET, budget_reason, response)

        if response.stop_reason == "max_tokens":
            if not state.response_recovery.has_escalated_tokens:
                state.response_recovery.max_tokens = ESCALATED_MAX_TOKENS
                state.response_recovery.has_escalated_tokens = True
                append_trace(
                    repo_root,
                    "max_tokens_recovery",
                    {"action": "escalate", "max_tokens": ESCALATED_MAX_TOKENS},
                )
                continue

            messages.append({"role": "assistant", "content": response.content})
            if state.response_recovery.continuation_count < MAX_CONTINUATIONS:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.response_recovery.continuation_count += 1
                append_trace(
                    repo_root,
                    "max_tokens_recovery",
                    {
                        "action": "continue",
                        "count": state.response_recovery.continuation_count,
                    },
                )
                continue
            append_trace(
                repo_root,
                "max_tokens_recovery",
                {"action": "stop", "count": state.response_recovery.continuation_count},
            )
            return finish(
                RunStatus.FAILED_BUDGET,
                "model output continuation budget exhausted",
                response,
            )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return finish(RunStatus.SUCCESS, "model completed without tool requests", response)

        results: list[dict[str, Any]] = []
        tool_blocks = [block for block in response.content if block_attr(block, "type") == "tool_use"]
        if not tool_blocks:
            return finish(RunStatus.FAILED_TOOL, "tool_use response contained no tool calls", response)
        for block in tool_blocks:
            call = parse_tool_call(block)
            fingerprint, progress_stop = state.progress.before_tool(call.name, call.arguments)
            if progress_stop is not None:
                return finish(progress_stop.status, progress_stop.reason, response)

            tool_result = execute_tool_call(
                call,
                handler=prepared.handlers.get(call.name),
                side_effect=tool_registry.has_side_effect(call.name),
                repo_root=repo_root,
                hook_registry=hook_registry,
                hook_context=hook_context,
                session_id=active_session_id,
                output=output,
            )
            tool_output = tool_result.to_model_content()
            state.metrics.tool_calls += 1
            if not tool_result.ok:
                state.metrics.tool_failures += 1
            progress_stop = state.progress.after_tool(fingerprint, tool_result)
            if progress_stop is not None:
                return finish(progress_stop.status, progress_stop.reason, response)

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.use_id,
                    "content": tool_output,
                }
            )

        # Anthropic 协议要求工具结果以 user 消息返回给模型。
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
    *,
    session_id: str = "",
    run_id: str | None = None,
) -> AgentRunResult:
    append_trace(
        repo_root,
        "agent_run_finished",
        {
            "session_id": session_id,
            "run_id": run_id,
            "status": status.value,
            "reason": reason,
            "metrics": metrics.to_dict(),
        },
    )
    return AgentRunResult(status=status, response=response, reason=reason, metrics=metrics)
