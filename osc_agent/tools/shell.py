from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
from typing import Literal

from pydantic import Field

from osc_agent.tools.process import CommandKind, CommandResult, classify_command, run_command
from osc_agent.tools.git import git_workspace_fingerprint
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

DEFAULT_TIMEOUT_SECONDS = 120
READ_ONLY_COMMANDS = {
    "get-childitem",
    "get-content",
    "get-location",
    "resolve-path",
    "select-string",
    "test-path",
    "rg",
}
READ_ONLY_GIT_COMMANDS = {"diff", "log", "show", "status"}
DENIED_SHELL_PATTERNS = (
    "remove-item -recurse",
    "format-volume",
    "stop-computer",
    "restart-computer",
)
DENIED_COMMAND_NAMES = {
    "cmd",
    "cmd.exe",
    "invoke-expression",
    "iex",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "start-process",
}
_POWERSHELL_AST_SCRIPT = r"""
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $env:OSC_AGENT_COMMAND,
    [ref]$tokens,
    [ref]$parseErrors
)
$commands = @(
    $ast.FindAll(
        { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
        $true
    ) | ForEach-Object {
        [pscustomobject]@{
            name = $_.GetCommandName()
            text = $_.Extent.Text
        }
    }
)
[pscustomobject]@{
    errors = @($parseErrors | ForEach-Object { $_.Message })
    commands = $commands
} | ConvertTo-Json -Compress -Depth 5
"""


class ShellInput(ContractModel):
    command: str = Field(min_length=1)
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=1_800)


class ShellOutput(ContractModel):
    command: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int = Field(ge=0)
    termination_reason: str | None = None
    command_kind: Literal["test", "build", "lint", "typecheck", "other"]
    workspace_fingerprint: str | None = None


class ShellTool(BaseTool[ShellInput, ShellOutput]):
    name = "powershell"
    description = "Run an approved PowerShell 7 command in the current repository directory."
    input_model = ShellInput
    output_model = ShellOutput

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("pwsh") or ""
        if not self.executable:
            raise ValueError("PowerShell 7 executable 'pwsh' was not found on PATH")

    def is_read_only(self, input: ShellInput) -> bool:
        return is_read_only_command(input.command)

    def is_concurrency_safe(self, input: ShellInput) -> bool:
        return self.is_read_only(input)

    def is_destructive(self, input: ShellInput) -> bool:
        return not self.is_read_only(input)

    def permission_risk(self, input: ShellInput) -> str:
        return "process"

    def permission_preview(
        self,
        input: ShellInput,
        context: ToolUseContext,
    ) -> dict[str, object]:
        return {
            "command": input.command,
            "timeout_seconds": input.timeout_seconds,
            "command_kind": classify_command(input.command).value,
        }

    async def validate_input(
        self,
        input: ShellInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        lowered = input.command.casefold()
        if pattern := next((item for item in DENIED_SHELL_PATTERNS if item in lowered), None):
            return ValidationFailure(reason=f"dangerous shell command contains {pattern!r}")
        try:
            analysis = await asyncio.to_thread(
                _analyze_powershell,
                input.command,
                executable=self.executable,
            )
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        errors = analysis.get("errors")
        if isinstance(errors, list) and errors:
            return ValidationFailure(reason=f"PowerShell parse failed: {errors[0]}")
        commands = analysis.get("commands")
        if not isinstance(commands, list) or not commands:
            return ValidationFailure(reason="PowerShell command contains no statically identifiable command")
        for command in commands:
            if not isinstance(command, dict):
                return ValidationFailure(reason="PowerShell command analysis returned invalid data")
            name = command.get("name")
            text = command.get("text")
            if not isinstance(name, str) or not name.strip():
                return ValidationFailure(reason="dynamic PowerShell command invocation is not allowed")
            canonical = Path(name.strip("\"'")).name.casefold()
            if canonical in DENIED_COMMAND_NAMES:
                return ValidationFailure(reason=f"nested or dynamic command runner is not allowed: {name}")
            if canonical in {"gh", "gh.exe"}:
                return ValidationFailure(reason="GitHub CLI commands are not allowed through PowerShell")
            if canonical in {"git", "git.exe"} and (
                not isinstance(text, str) or not is_read_only_command(text)
            ):
                return ValidationFailure(
                    reason="Git write operations are not allowed through PowerShell; use structured Git tools"
                )
        return ValidationSuccess()

    async def check_permissions(
        self,
        input: ShellInput,
        context: ToolUseContext,
    ) -> PermissionDecision:
        # 非只读命令已由通用 destructive permission 完成一次授权。
        return Allow(updated_input=input.model_dump(mode="json"))

    async def call(self, input: ShellInput, context: ToolUseContext) -> ToolResult:
        result = await run_powershell(
            input.command,
            executable=self.executable,
            repo_root=Path(context.working_directory),
            timeout_seconds=input.timeout_seconds,
            enforce_risk_checks=False,
        )
        if result.termination_reason in {"timeout", "os_error"}:
            return ToolResult(
                error=ToolError(
                    code="POWERSHELL_TIMEOUT"
                    if result.termination_reason == "timeout"
                    else "POWERSHELL_START_FAILED",
                    message=result.output or result.termination_reason,
                    retryable=result.termination_reason == "timeout",
                )
            )
        try:
            fingerprint = await asyncio.to_thread(
                git_workspace_fingerprint,
                repo_root=Path(context.working_directory),
            )
        except (OSError, ValueError):
            fingerprint = None
        return ToolResult(
            data={
                "command": result.command,
                "success": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "termination_reason": result.termination_reason,
                "command_kind": classify_command(input.command).value,
                "workspace_fingerprint": fingerprint,
            }
        )


def is_read_only_command(command: str) -> bool:
    if any(marker in command for marker in ("&&", "||", ";", "|", ">", "<", "`", "$", "&", "\r", "\n")):
        return False
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', command)
    if " ".join(tokens).strip() != command.strip():
        return False
    if not tokens:
        return False
    executable = Path(tokens[0].strip("\"'")).stem.casefold()
    arguments = [token.strip("\"'") for token in tokens[1:]]
    if any(_argument_escapes_repository(argument) for argument in arguments):
        return False
    if executable == "rg" and any(
        argument == "--pre" or argument.startswith("--pre=") for argument in arguments
    ):
        return False
    if executable == "git" and any(
        argument == "--output" or argument.startswith("--output=") for argument in arguments
    ):
        return False
    if executable in READ_ONLY_COMMANDS:
        return True
    return (
        executable == "git"
        and len(tokens) >= 2
        and tokens[1].strip("\"'").casefold() in READ_ONLY_GIT_COMMANDS
    )


def _argument_escapes_repository(argument: str) -> bool:
    candidate = argument.split("=", 1)[-1]
    if PureWindowsPath(candidate).is_absolute() or PurePosixPath(candidate).is_absolute():
        return True
    normalized = candidate.replace("\\", "/")
    return ".." in PurePosixPath(normalized).parts


async def run_powershell(
    command: str,
    *,
    executable: str,
    repo_root: Path,
    timeout_seconds: int | float = DEFAULT_TIMEOUT_SECONDS,
    enforce_risk_checks: bool = True,
) -> CommandResult:
    """在目标 repo 内执行命令，并统一处理超时、空输出和长度截断。"""
    if enforce_risk_checks:
        lowered = command.casefold()
        if pattern := next((item for item in DENIED_SHELL_PATTERNS if item in lowered), None):
            return CommandResult(
                command=command,
                exit_code=-2,
                stderr=f"Permission denied: dangerous shell command contains {pattern!r}",
                duration_ms=0,
                termination_reason="permission_denied",
            )

    result = await run_command(
        executable,
        command,
        repo_root=repo_root,
        timeout_seconds=timeout_seconds,
    )
    return result


def _analyze_powershell(command: str, *, executable: str) -> dict[str, object]:
    env = {**os.environ, "OSC_AGENT_COMMAND": command}
    try:
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _POWERSHELL_AST_SCRIPT,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"PowerShell command analysis failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown parser failure").strip()
        raise ValueError(f"PowerShell command analysis failed: {detail}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("PowerShell command analysis returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("PowerShell command analysis returned an invalid payload")
    return value
