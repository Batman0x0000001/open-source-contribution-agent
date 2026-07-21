"""
主 Agent 调用 subagent
    ↓
spawn_subagent()
    ↓
创建全新的 messages
    ↓
子 Agent 使用只读工具分析
    ↓
多轮 tool_use / tool_result
    ↓
子 Agent 给出最终 summary
    ↓
summary 返回给主 Agent
"""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from time import perf_counter
from typing import Any, Callable

from osc_agent.config import Settings
from osc_agent.harness.hooks import HookContext, default_hooks, elapsed_ms
from osc_agent.harness.trace import append_trace, preview
from osc_agent.tools.files import FILE_TOOLS, glob_files, read_file
from osc_agent.tools.git import GIT_TOOLS, git_status
from osc_agent.tools.repo import REPO_TOOLS, inspect_repo
from osc_agent.tools.shell import BASH_TOOL

SUBAGENT_ROLES = {"issue_analyzer", "repo_mapper", "test_analyzer", "doc_reviewer"}
SUBAGENT_MAX_ROUNDS = 30
READ_ONLY_EXECUTABLES = ("findstr", "git", "grep", "rg")
READ_ONLY_GIT_SUBCOMMANDS = {"diff", "log", "show", "status"}
READ_ONLY_COMMAND_TIMEOUT_SECONDS = 120
READ_ONLY_OUTPUT_LIMIT = 50_000

SUBAGENT_TOOL = {
    "name": "subagent",
    "description": "Launch a read-only subagent with fresh context and return only its final summary.",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "role": {
                "type": "string",
                "enum": sorted(SUBAGENT_ROLES),
            },
        },
        "required": ["description", "role"],
        "additionalProperties": False,
    },
}

SUBAGENT_TOOLS = [
    BASH_TOOL,
    FILE_TOOLS[0],
    FILE_TOOLS[3],
    GIT_TOOLS[0],
    REPO_TOOLS[0],
]


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _tool_input(block: Any) -> dict[str, Any]:
    value = _block_attr(block, "input", {})
    return value if isinstance(value, dict) else {}


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = _block_attr(block, "text")
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return str(content)


def run_read_only_bash(command: str, *, repo_root: Path) -> str:
    """以参数数组执行受限查询命令，禁止 shell 拼接、重定向和仓库外路径。"""
    if any(marker in command for marker in ("&", "|", ";", "<", ">", "`", "\r", "\n")):
        return "Permission denied: shell operators are not allowed in read-only commands"
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        return f"Permission denied: invalid read-only command: {exc}"
    if not argv or argv[0].lower() not in READ_ONLY_EXECUTABLES:
        return "Permission denied: subagent bash is read-only"

    executable = argv[0].lower()
    if executable == "git":
        if len(argv) < 2 or argv[1].lower() not in READ_ONLY_GIT_SUBCOMMANDS:
            return "Permission denied: only git diff/log/show/status are read-only"
        if any(arg == "--output" or arg.startswith("--output=") for arg in argv[2:]):
            return "Permission denied: git output files are not allowed"
        if argv[1].lower() == "diff":
            argv[2:2] = ["--no-ext-diff", "--no-textconv"]
        elif argv[1].lower() in {"log", "show"}:
            argv.insert(2, "--no-textconv")
    elif executable == "rg":
        if any(arg == "--pre" or arg.startswith("--pre=") for arg in argv[1:]):
            return "Permission denied: rg preprocessors are not allowed"
        argv.insert(1, "--no-config")

    if any(_argument_escapes_repo(arg) for arg in argv[1:]):
        return "Permission denied: read-only command path escapes repository"

    try:
        completed = subprocess.run(
            argv,
            shell=False,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=READ_ONLY_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: read-only command timed out after {READ_ONLY_COMMAND_TIMEOUT_SECONDS} seconds"
    except OSError as exc:
        return f"Error: read-only command failed: {exc}"

    output = ((completed.stdout or "") + (completed.stderr or "")).strip() or "(no output)"
    if completed.returncode != 0:
        output = f"Error: read-only command exited with code {completed.returncode}\n{output}"
    return output[:READ_ONLY_OUTPUT_LIMIT]


def _argument_escapes_repo(argument: str) -> bool:
    candidate = argument.split("=", 1)[-1]
    path = Path(candidate)
    return path.is_absolute() or ".." in path.parts


def _subagent_handlers(repo_root: Path) -> dict[str, Any]:
    """子 agent 只拿到只读工具；不暴露 subagent，防止递归委派。"""
    return {
        "bash": lambda command: run_read_only_bash(command, repo_root=repo_root),
        "read_file": lambda path, limit=20_000, offset=0: read_file(
            repo_root=repo_root,
            path=path,
            limit=limit,
            offset=offset,
        ),
        "glob": lambda pattern: glob_files(repo_root=repo_root, pattern=pattern),
        "git_status": lambda: git_status(repo_root=repo_root),
        "inspect_repo": lambda: inspect_repo(repo_root=repo_root),
    }


def spawn_subagent(
    description: str,
    role: str,
    *,
    client: Any,
    settings: Settings,
    repo_root: Path,
    confirm: Callable[[str], bool] | None = None,
) -> str:
    """用全新 messages 跑只读子 agent，并只把最终摘要交回主上下文。"""
    if role not in SUBAGENT_ROLES:
        raise ValueError(f"role must be one of: {', '.join(sorted(SUBAGENT_ROLES))}")
    if not description.strip():
        raise ValueError("description must be a non-empty string")

    append_trace(repo_root, "subagent_start", {"agent": "subagent", "role": role, "description": description})

    messages: list[dict[str, Any]] = [{"role": "user", "content": description}]
    handlers = _subagent_handlers(repo_root)
    hook_registry = default_hooks()
    hook_context = HookContext(repo_root=repo_root, confirm=confirm)
    system_prompt = (
        f"You are a read-only {role} subagent. Analyze the task using only the provided read-only tools. "
        "Do not delegate. Return a concise structured summary with findings and evidence."
    )
    final_summary = ""

    for _round_index in range(1, SUBAGENT_MAX_ROUNDS + 1):
        response = client.messages.create(
            model=settings.model_id,
            system=system_prompt,
            messages=messages,
            tools=SUBAGENT_TOOLS,
            max_tokens=4000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_summary = _extract_text(response.content)
            break

        results: list[dict[str, str]] = []
        for block in response.content:
            if _block_attr(block, "type") != "tool_use":
                continue

            tool_name = _block_attr(block, "name")
            tool_use_id = _block_attr(block, "id")
            tool_args = _tool_input(block)
            handler = handlers.get(tool_name)
            pre_results = hook_registry.run(
                "PreToolUse",
                hook_context,
                {"tool_name": tool_name, "tool_args": tool_args},
            )
            blocked = next((result for result in pre_results if not result.allowed), None)
            started = perf_counter()
            if blocked is not None:
                tool_output = blocked.content or "Permission denied"
            elif handler is None:
                tool_output = f"Error: unknown subagent tool {tool_name}"
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
            append_trace(
                repo_root,
                "subagent_tool_use",
                {
                    "agent": "subagent",
                    "role": role,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "output_preview": preview(tool_output),
                },
            )
            results.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": tool_output})

        messages.append({"role": "user", "content": results})
    else:
        final_summary = f"Subagent stopped after {SUBAGENT_MAX_ROUNDS} rounds without a final answer."

    summary = f"role: {role}\ndescription: {description}\nsummary:\n{final_summary}".strip()
    append_trace(repo_root, "subagent_stop", {"agent": "subagent", "role": role, "summary": preview(summary, 1000)})
    return summary
