"""提供受权限和资源策略约束的 Bash 命令工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import shlex
import shutil
from typing import Literal

from pydantic import Field

from osc_agent.runtime.models import (
    Allow,
    ContractModel,
    PermissionDecision,
    ToolError,
    ToolResult,
    ToolUseContext,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.tools.git import git_workspace_fingerprint
from osc_agent.tools.process_runner import (
    HostProcessRunner,
    ProcessRunner,
    ProcessRequest,
    build_subprocess_environment,
    classify_command,
)


DEFAULT_TIMEOUT_SECONDS = 120
READ_ONLY_COMMANDS = {"rg"}
READ_ONLY_GIT_COMMANDS = {"diff", "log", "show", "status"}
_HARD_DENY = (
    re.compile(r"(^|[;&|]\s*)\s*(sudo|su|mount|umount|shutdown|reboot|systemctl)\b", re.I),
    re.compile(r"\brm\s+(?:-[A-Za-z]*r[A-Za-z]*f|-[A-Za-z]*f[A-Za-z]*r)\s+/(?:\s|$)", re.I),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:", re.I),
)


class BashInput(ContractModel):
    command: str = Field(min_length=1)
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=1_800)


class BashOutput(ContractModel):
    command: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int = Field(ge=0)
    termination_reason: str | None = None
    command_kind: Literal["test", "build", "lint", "typecheck", "other"]
    workspace_fingerprint: str | None = None


class BashTool(BaseTool[BashInput, BashOutput]):
    name = "bash"
    description = "Run an approved non-interactive Bash command in the current repository directory."
    input_model = BashInput
    output_model = BashOutput

    def __init__(
        self,
        executable: str | None = None,
        *,
        environment_allowlist: frozenset[str] = frozenset(),
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.process_runner = process_runner or HostProcessRunner()
        self.executable = executable or shutil.which("bash") or "/bin/bash"
        self.environment = build_subprocess_environment(environment_allowlist)

    def is_read_only(self, input: BashInput) -> bool:
        return is_read_only_command(input.command)

    def is_concurrency_safe(self, input: BashInput) -> bool:
        return self.is_read_only(input)

    def is_destructive(self, input: BashInput) -> bool:
        return not self.is_read_only(input)

    def permission_risk(self, input: BashInput) -> str:
        return "process"

    def permission_preview(self, input: BashInput, context: ToolUseContext) -> dict[str, object]:
        return {
            "command": input.command,
            "timeout_seconds": input.timeout_seconds,
            "command_kind": classify_command(input.command).value,
        }

    async def validate_input(self, input: BashInput, context: ToolUseContext) -> ValidationResult:
        if "\x00" in input.command or any(pattern.search(input.command) for pattern in _HARD_DENY):
            return ValidationFailure(reason="Bash command violates the hard host-safety policy")
        try:
            shlex.split(input.command, posix=True)
        except ValueError as exc:
            return ValidationFailure(reason=f"Bash parse failed: {exc}")
        return ValidationSuccess()

    async def check_permissions(
        self, input: BashInput, context: ToolUseContext
    ) -> PermissionDecision:
        return Allow(updated_input=input.model_dump(mode="json"))

    async def call(self, input: BashInput, context: ToolUseContext) -> ToolResult:
        result = await self.process_runner.run(
            ProcessRequest(
                executable=self.executable,
                command=input.command,
                repo_root=context.working_directory,
                timeout_seconds=input.timeout_seconds,
                environment=self.environment,
            ),
            context,
        )
        if result.termination_reason in {"timeout", "os_error", "cancelled"}:
            return ToolResult(error=ToolError(
                code={
                    "timeout": "BASH_TIMEOUT",
                    "os_error": "BASH_START_FAILED",
                    "cancelled": "BASH_CANCELLED",
                }[result.termination_reason],
                message=result.output or result.termination_reason,
                retryable=result.termination_reason == "timeout",
            ))
        try:
            fingerprint = await asyncio.to_thread(
                git_workspace_fingerprint, repo_root=Path(context.working_directory)
            )
        except (OSError, ValueError):
            fingerprint = None
        return ToolResult(data={
            "command": result.command,
            "success": result.exit_code == 0,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
            "termination_reason": result.termination_reason,
            "command_kind": classify_command(input.command).value,
            "workspace_fingerprint": fingerprint,
        })


def is_read_only_command(command: str) -> bool:
    if any(marker in command for marker in ("&&", "||", ";", "|", ">", "<", "`", "$", "&", "\r", "\n")):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name.casefold()
    if executable == "rg":
        return not any(arg in {"--pre", "--follow", "-L"} or arg.startswith("--pre=") for arg in tokens[1:])
    return executable == "git" and len(tokens) >= 2 and tokens[1].casefold() in READ_ONLY_GIT_COMMANDS
