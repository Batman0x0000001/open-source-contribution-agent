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
from pathlib import Path

from osc_agent.harness.command import CommandKind, classify_command, run_command
from osc_agent.harness.risk import assess_shell_risk, format_risk_block

BASH_TOOL = {
    "name": "bash",
    "description": "Run a shell command inside the target repository.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run from the repository root.",
            },
            "run_in_background": {
                "type": "boolean",
                "description": (
                    "Set true only for slow commands such as pytest, npm test, cargo test, or build commands "
                    "when the agent should continue working while the command runs."
                ),
            },
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
    enforce_risk_checks: bool = True,
) -> str:
    """在目标 repo 内执行命令，并统一处理超时、空输出和长度截断。"""
    if enforce_risk_checks:
        decision = assess_shell_risk(command)
        if not decision.allowed:
            return format_risk_block(decision)

    result = run_command(command, repo_root=repo_root, timeout_seconds=timeout_seconds)
    if result.termination_reason in {"timeout", "os_error"}:
        return _structured_error(result.termination_reason, result.output)

    output = result.output
    output = output.strip() or "(no output)"
    if result.exit_code != 0:
        output = (
            _structured_error("nonzero_exit", f"command exited with code {result.exit_code}")
            + "\n"
            + output
        )
        if classify_command(command) is CommandKind.TEST:
            output += (
                "\n\nRecovery guidance: tests failed. Read the failure summary, locate related files, "
                "update todo status, then rerun the narrowest relevant test."
            )

    return output[:MAX_OUTPUT_CHARS]


def _structured_error(kind: str, message: str) -> str:
    """把 shell 错误转成稳定文本结构，方便 agent 按 kind 做恢复判断。"""
    return "Error: " + json.dumps({"kind": kind, "message": message}, ensure_ascii=False)
