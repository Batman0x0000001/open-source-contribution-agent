from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil

from pydantic import Field

from osc_agent.tools.process import CommandKind, classify_command, run_command
from osc_agent.runtime.models import (
    Allow,
    ContractModel,
    PermissionDecision,
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
    "git push",
    "gh pr create",
    "remove-item -recurse",
    "format-volume",
    "stop-computer",
    "restart-computer",
)


class ShellInput(ContractModel):
    command: str = Field(min_length=1)
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=1_800)


class ShellOutput(ContractModel):
    output: str


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

    async def validate_input(
        self,
        input: ShellInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        lowered = input.command.casefold()
        if pattern := next((item for item in DENIED_SHELL_PATTERNS if item in lowered), None):
            return ValidationFailure(reason=f"dangerous shell command contains {pattern!r}")
        return ValidationSuccess()

    async def check_permissions(
        self,
        input: ShellInput,
        context: ToolUseContext,
    ) -> PermissionDecision:
        # 非只读命令已由通用 destructive permission 完成一次授权。
        return Allow(updated_input=input.model_dump(mode="json"))

    async def call(self, input: ShellInput, context: ToolUseContext) -> ToolResult:
        output = await run_powershell(
            input.command,
            executable=self.executable,
            repo_root=Path(context.working_directory),
            timeout_seconds=input.timeout_seconds,
            enforce_risk_checks=False,
        )
        return ToolResult(data={"output": output})


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
) -> str:
    """在目标 repo 内执行命令，并统一处理超时、空输出和长度截断。"""
    if enforce_risk_checks:
        lowered = command.casefold()
        if pattern := next((item for item in DENIED_SHELL_PATTERNS if item in lowered), None):
            return f"Permission denied: dangerous shell command contains {pattern!r}"

    result = await run_command(
        executable,
        command,
        repo_root=repo_root,
        timeout_seconds=timeout_seconds,
    )
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
                "fix the narrowest supported cause, then rerun the narrowest relevant test."
            )

    return output


def _structured_error(kind: str, message: str) -> str:
    """把 shell 错误转成稳定文本结构，方便 agent 按 kind 做恢复判断。"""
    return "Error: " + json.dumps({"kind": kind, "message": message}, ensure_ascii=False)
