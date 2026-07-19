"""
agent_loop 准备调用工具
        ↓
PreToolUse hook：能力边界 → 仓库边界 → 风险评估 → 审批
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

from osc_agent.harness.capabilities import AgentCapabilityScope
from osc_agent.harness.repository_boundary import safe_repo_path
from osc_agent.harness.risk import RiskDecision, assess_file_write_risk, assess_shell_risk, format_risk_block
from osc_agent.harness.trace import append_trace, preview, sanitize_tool_arguments


@dataclass
class HookContext:
    repo_root: Path
    capabilities: AgentCapabilityScope = field(default_factory=AgentCapabilityScope.unrestricted)
    session_id: str = ""
    run_id: str | None = None
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
    registry.register("PreToolUse", pre_tool_guard_hook)
    registry.register("PostToolUse", post_tool_trace_hook)
    registry.register("Stop", stop_summary_hook)
    return registry


def pre_tool_guard_hook(context: HookContext, payload: dict[str, Any]) -> PreToolUseResult:
    """按能力、路径和风险层次检查工具调用，并在需要时请求审批。"""
    tool_name = str(payload.get("tool_name", ""))
    tool_args = payload.get("tool_args", {})
    if not isinstance(tool_args, dict):
        return PreToolUseResult(False, "Error: tool arguments must be an object")

    capability_violation = _capability_violation(tool_name, tool_args, context.capabilities)
    if capability_violation is not None:
        _record_guard_decision(context, tool_name, "capability", "deny", capability_violation)
        return PreToolUseResult(False, f"Permission denied: {capability_violation}")

    boundary_violation = _repository_boundary_violation(context.repo_root, tool_name, tool_args)
    if boundary_violation is not None:
        _record_guard_decision(context, tool_name, "repository_boundary", "deny", boundary_violation)
        return PreToolUseResult(False, f"Permission denied: {boundary_violation}")

    decision = _assess_tool_risk(tool_name, tool_args)
    _record_guard_decision(context, tool_name, "risk", decision.action, decision.reason)
    if decision.allowed:
        return PreToolUseResult(True)
    if decision.action == "ask":
        if context.confirm is None:
            return PreToolUseResult(False, format_risk_block(decision))

        # 风险层只提出审批要求，是否放行由 Hook 中的确认结果决定。
        approved = context.confirm(f"{decision.reason} Allow this tool call?")
        append_trace(
            context.repo_root,
            "approval_decision",
            {
                "session_id": context.session_id,
                "run_id": context.run_id,
                "tool": tool_name,
                "approved": approved,
                "reason": decision.reason,
            },
        )
        if approved:
            return PreToolUseResult(True)
    return PreToolUseResult(False, format_risk_block(decision))


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
            "session_id": context.session_id,
            "run_id": context.run_id,
            "stage": context.capabilities.stage.value,
            "tool": tool_name,
            "arguments": sanitize_tool_arguments(tool_name, tool_args),
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
            "session_id": context.session_id,
            "run_id": context.run_id,
            "stage": context.capabilities.stage.value,
            "tool_count": context.tool_count,
            "failed_count": context.failed_count,
            "modified_files": sorted(context.modified_files),
            "stop_reason": payload.get("stop_reason"),
        },
    )


def _capability_violation(
    tool_name: str,
    tool_args: dict[str, Any],
    capabilities: AgentCapabilityScope,
) -> str | None:
    if not capabilities.permits_tool(tool_name):
        return f"tool {tool_name} is not allowed during {capabilities.stage.value}"
    if tool_name in {"write_file", "edit_file"} and not capabilities.permits_write(
        str(tool_args.get("path", ""))
    ):
        return f"{tool_name.removesuffix('_file')} path is outside the current stage scope"
    return None


def _repository_boundary_violation(
    repo_root: Path,
    tool_name: str,
    tool_args: dict[str, Any],
) -> str | None:
    if tool_name not in {"read_file", "write_file", "edit_file"}:
        return None
    try:
        safe_repo_path(repo_root, str(tool_args.get("path", "")))
    except ValueError as exc:
        return str(exc)
    return None


def _assess_tool_risk(tool_name: str, tool_args: dict[str, Any]) -> RiskDecision:
    if tool_name == "bash":
        return assess_shell_risk(str(tool_args.get("command", "")))
    if tool_name == "write_file":
        return assess_file_write_risk(
            str(tool_args.get("path", "")),
            str(tool_args.get("content", "")),
        )
    if tool_name == "edit_file":
        return assess_file_write_risk(
            str(tool_args.get("path", "")),
            str(tool_args.get("new_text", "")),
        )
    return RiskDecision("allow", "tool risk accepted")


def _record_guard_decision(
    context: HookContext,
    tool_name: str,
    layer: str,
    action: str,
    reason: str,
) -> None:
    append_trace(
        context.repo_root,
        "guard_decision",
        {
            "session_id": context.session_id,
            "run_id": context.run_id,
            "stage": context.capabilities.stage.value,
            "tool": tool_name,
            "layer": layer,
            "action": action,
            "reason": reason,
        },
    )


def elapsed_ms(start: float) -> int:
    """把 perf_counter 起点转换为毫秒，避免 agent_loop 里散落计时细节。"""
    return int((perf_counter() - start) * 1000)
