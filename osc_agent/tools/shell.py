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

import json
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
    enforce_permissions: bool = True,
) -> str:
    """在目标 repo 内执行命令，并统一处理超时、空输出和长度截断。"""
    if enforce_permissions:
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
        return _structured_error("timeout", f"command timed out after {timeout_seconds} seconds")
    except OSError as exc:
        return _structured_error("os_error", str(exc))

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip() or "(no output)"
    if completed.returncode != 0:
        output = (
            _structured_error("nonzero_exit", f"command exited with code {completed.returncode}")
            + "\n"
            + output
        )
        if _looks_like_test_command(command):
            output += (
                "\n\nRecovery guidance: tests failed. Read the failure summary, locate related files, "
                "update todo status, then rerun the narrowest relevant test."
            )

    return output[:MAX_OUTPUT_CHARS]


def _structured_error(kind: str, message: str) -> str:
    """把 shell 错误转成稳定文本结构，方便 agent 按 kind 做恢复判断。"""
    return "Error: " + json.dumps({"kind": kind, "message": message}, ensure_ascii=False)


def _looks_like_test_command(command: str) -> bool:
    lowered = command.lower()
    return any(marker in lowered for marker in ("pytest", "npm test", "pnpm test", "yarn test", "cargo test"))
