"""
模型请求 bash
      ↓
agent_loop 调用 run_bash()
      ↓
subprocess.run() 执行命令
      ↓
获取 stdout + stderr
      ↓
处理异常、空输出、长度限制
      ↓
返回字符串给 Agent
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from osc_agent.harness.permissions import check_shell_command, format_blocked

BASH_TOOL = {
    "name": "bash",
    "description": "Run a shell command inside the target repository.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run from the repository root.",
            }
        },
        "required": ["command"],
    },
}

MAX_OUTPUT_CHARS = 50_000
DEFAULT_TIMEOUT_SECONDS = 120


def run_bash(
    command: str,
    *,
    repo_root: Path,
    timeout_seconds: int | float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """在目标 repo 内执行命令，并统一处理超时、空输出和长度截断。"""
    decision = check_shell_command(command)
    if not decision.allowed:
        return format_blocked(decision)

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout_seconds} seconds"
    except OSError as exc:
        return f"Error: {exc}"

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip() or "(no output)"

    return output[:MAX_OUTPUT_CHARS]
