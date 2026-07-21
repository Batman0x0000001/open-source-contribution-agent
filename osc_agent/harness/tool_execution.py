from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TextIO

from osc_agent.harness.background import should_run_background, start_background_task
from osc_agent.harness.contracts import ToolResult, normalize_tool_result
from osc_agent.harness.hooks import HookContext, HookRegistry, elapsed_ms
from osc_agent.harness.runtime_state import record_tool_observation


@dataclass(frozen=True)
class ToolCall:
    name: str
    use_id: str | None
    arguments: dict[str, Any]


def block_attr(block: Any, name: str, default: Any = None) -> Any:
    """兼容 Anthropic SDK 对象和测试使用的字典 block。"""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def parse_tool_call(block: Any) -> ToolCall:
    raw_arguments = block_attr(block, "input", {})
    return ToolCall(
        name=str(block_attr(block, "name", "")),
        use_id=(str(value) if (value := block_attr(block, "id")) is not None else None),
        arguments=raw_arguments if isinstance(raw_arguments, dict) else {},
    )


def execute_tool_call(
    call: ToolCall,
    *,
    handler: Callable[..., Any] | None,
    side_effect: bool,
    repo_root: Path,
    hook_registry: HookRegistry,
    hook_context: HookContext,
    session_id: str,
    output: TextIO | None = None,
) -> ToolResult:
    """执行一次工具调用，并统一完成权限、后台调度、规范化和审计。"""
    started = perf_counter()
    if handler is None:
        raw_output: Any = f"Error: unknown tool {call.name}"
    else:
        if output is not None:
            print(f"{call.name}: {call.arguments}", file=output)
        pre_results = hook_registry.run(
            "PreToolUse",
            hook_context,
            {"tool_name": call.name, "tool_args": call.arguments},
        )
        blocked = next((result for result in pre_results if not result.allowed), None)
        if blocked is not None:
            raw_output = blocked.content or "Permission denied"
        elif should_run_background(call.name, call.arguments):
            # 后台任务的真实结果由下一轮通知注入，当前调用只返回任务标识。
            foreground_args = dict(call.arguments)
            foreground_args.pop("run_in_background", None)
            task_id = start_background_task(
                command=str(call.arguments.get("command", "")),
                repo_root=repo_root,
                runner=lambda: handler(**foreground_args),
            )
            raw_output = (
                f"[Background task {task_id} started] "
                "Result will be available in a later task_notification."
            )
        else:
            try:
                raw_output = handler(**call.arguments)
            except (TypeError, ValueError) as exc:
                raw_output = f"Error: invalid arguments for {call.name}: {exc}"
            except Exception as exc:  # noqa: BLE001 - 工具异常必须转换为可审计结果。
                raw_output = f"Error: tool {call.name} failed: {exc}"

    result = normalize_tool_result(
        raw_output,
        tool_name=call.name,
        arguments=call.arguments,
        call_id=call.use_id,
        latency_ms=elapsed_ms(started),
        side_effect=side_effect,
    )
    record_tool_observation(
        repo_root,
        call.name,
        call.arguments,
        result,
        session_id=session_id,
    )
    hook_registry.run(
        "PostToolUse",
        hook_context,
        {
            "tool_name": call.name,
            "tool_args": call.arguments,
            "output": result.to_dict(),
            "latency_ms": result.latency_ms,
        },
    )
    if output is not None:
        print(result.to_model_content()[:200], file=output)
    return result
