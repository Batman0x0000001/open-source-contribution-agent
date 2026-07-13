"""
agent_loop 准备调用工具
        ↓
PreToolUse hook：权限检查
        ↓
允许 → 执行工具
拒绝 → 返回错误文本
        ↓
PostToolUse hook：记录工具调用日志
        ↓
模型最终停止
        ↓
Stop hook：记录本轮摘要
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from osc_agent.harness.permissions import (
    PermissionDecision,
    allow,
    check_file_write,
    check_shell_command,
    format_blocked,
    safe_repo_path,
)
from osc_agent.harness.trace import append_trace, preview


@dataclass
class HookContext:
    repo_root: Path
    confirm: Callable[[str], bool] | None = None
    tool_count: int = 0
    failed_count: int = 0
    modified_files: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PreToolUseResult:
    allowed: bool
    content: str | None = None


HookHandler = Callable[[HookContext, dict[str, Any]], PreToolUseResult | None]


class HookRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, list[HookHandler]] = {
            "UserPromptSubmit": [],
            "PreToolUse": [],
            "PostToolUse": [],
            "Stop": [],
        }

    def register(self, event: str, handler: HookHandler) -> None:
        """注册 hook 处理器，后续阶段可以继续扩展事件行为。"""
        self._handlers.setdefault(event, []).append(handler)

    def extend(self, other: "HookRegistry") -> None:
        """合并外部 hooks，但不替换默认安全 hooks。"""
        for event, handlers in other._handlers.items():
            self._handlers.setdefault(event, []).extend(handlers)

    def run(self, event: str, context: HookContext, payload: dict[str, Any]) -> list[PreToolUseResult]:
        """按注册顺序执行 hook；PreToolUse 可以返回阻止结果。"""
        results: list[PreToolUseResult] = []
        #此时的handler就是HookHandler，也就是函数，default为None
        for handler in self._handlers.get(event, []):
            result = handler(context, payload)
            if result is not None:
                results.append(result)
        return results


def default_hooks() -> HookRegistry:
    registry = HookRegistry()
    registry.register("PreToolUse", pre_tool_permission_hook)
    registry.register("PostToolUse", post_tool_trace_hook)
    registry.register("Stop", stop_summary_hook)
    return registry


def pre_tool_permission_hook(context: HookContext, payload: dict[str, Any]) -> PreToolUseResult:
    """在工具执行前集中做权限判断，阻止危险操作进入 handler。"""
    tool_name = str(payload.get("tool_name", ""))
    tool_args = payload.get("tool_args", {})
    if not isinstance(tool_args, dict):
        return PreToolUseResult(False, "Error: tool arguments must be an object")

    decision = _permission_for_tool(context.repo_root, tool_name, tool_args)
    append_trace(
        context.repo_root,
        "permission_decision",
        {
            "tool": tool_name,
            "action": decision.action,
            "reason": decision.reason,
        },
    )
    if decision.allowed:
        return PreToolUseResult(True)
    if decision.action == "ask":
        if context.confirm is None:
            return PreToolUseResult(False, format_blocked(decision))

        # ask 决策必须交给人类确认；只有明确输入 y/yes 才放行。
        approved = context.confirm(f"{decision.reason} Allow this tool call?")
        append_trace(
            context.repo_root,
            "permission_confirmation",
            {
                "tool": tool_name,
                "approved": approved,
                "reason": decision.reason,
            },
        )
        if approved:
            return PreToolUseResult(True)
    return PreToolUseResult(False, format_blocked(decision))


def post_tool_trace_hook(context: HookContext, payload: dict[str, Any]) -> None:
    """工具执行后记录名称、参数、输出预览、耗时和错误状态。"""
    context.tool_count += 1
    raw_output = payload.get("output", "")
    output = str(raw_output)
    if isinstance(raw_output, dict) and "ok" in raw_output:
        failed = not bool(raw_output.get("ok"))
    else:
        failed = output.startswith(("Error:", "Permission denied:", "Permission required:", "Git error:"))
    if failed:
        context.failed_count += 1

    tool_name = str(payload.get("tool_name", ""))
    tool_args = payload.get("tool_args", {})
    if tool_name in {"write_file", "edit_file"} and isinstance(tool_args, dict):
        path = tool_args.get("path")
        if isinstance(path, str):
            context.modified_files.add(path)

    append_trace(
        context.repo_root,
        "tool_use",
        {
            "tool": tool_name,
            "arguments": tool_args,
            "output_preview": preview(output),
            "latency_ms": payload.get("latency_ms", 0),
            "error": failed,
        },
    )


def stop_summary_hook(context: HookContext, payload: dict[str, Any]) -> None:
    """模型停止时写入本轮摘要，方便审计本轮工具活动。"""
    append_trace(
        context.repo_root,
        "stop_summary",
        {
            "tool_count": context.tool_count,
            "failed_count": context.failed_count,
            "modified_files": sorted(context.modified_files),
            "stop_reason": payload.get("stop_reason"),
        },
    )


def _permission_for_tool(repo_root: Path, tool_name: str, tool_args: dict[str, Any]) -> PermissionDecision:
    if tool_name == "bash":
        return check_shell_command(str(tool_args.get("command", "")))

    if tool_name in {"read_file", "write_file", "edit_file"}:
        path = str(tool_args.get("path", ""))
        try:
            safe_repo_path(repo_root, path)
        except ValueError as exc:
            return PermissionDecision("deny", str(exc))

    if tool_name == "write_file":
        return check_file_write(str(tool_args.get("path", "")), str(tool_args.get("content", "")))

    if tool_name == "edit_file":
        return check_file_write(str(tool_args.get("path", "")), str(tool_args.get("new_text", "")))

    return allow("tool allowed")


def elapsed_ms(start: float) -> int:
    """把 perf_counter 起点转换为毫秒，避免 agent_loop 里散落计时细节。"""
    return int((perf_counter() - start) * 1000)
