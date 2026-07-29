"""提供基于 ripgrep 的仓库文本搜索工具。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import subprocess

from pydantic import Field

from osc_agent.runtime.instructions import RepositoryInstructionResolver
from osc_agent.runtime.models import (
    ContextUpdate,
    ContractModel,
    ToolError,
    ToolResult,
    ToolUseContext,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.processes.policy import build_subprocess_environment
from osc_agent.workspaces.path_policy import normalize_repo_relative_path, safe_repo_path


class GrepInput(ContractModel):
    pattern: str = Field(min_length=1, description="Regular expression to search for.")
    path: str = Field(default=".", description="Repository-relative file or directory.")
    glob: str | None = Field(default=None, description="Optional ripgrep file glob.")
    case_sensitive: bool = True
    context_lines: int = Field(default=0, ge=0, le=20)
    max_matches: int = Field(default=200, ge=1, le=2_000)


class GrepMatch(ContractModel):
    path: str
    line: int = Field(ge=1)
    text: str


class GrepOutput(ContractModel):
    matches: list[GrepMatch]
    truncated: bool


class GrepTool(BaseTool[GrepInput, GrepOutput]):
    name = "grep"
    description = "Search repository text with ripgrep and return bounded structured matches."
    input_model = GrepInput
    output_model = GrepOutput

    def __init__(
        self,
        instructions: RepositoryInstructionResolver | None = None,
        executable: str | None = None,
    ) -> None:
        self.instructions = instructions or RepositoryInstructionResolver()
        self.executable = executable or shutil.which("rg") or ""
        if not self.executable:
            raise ValueError("ripgrep executable 'rg' was not found on PATH")

    def is_read_only(self, input: GrepInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GrepInput) -> bool:
        return True

    async def validate_input(
        self,
        input: GrepInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        try:
            if input.path != ".":
                normalize_repo_relative_path(input.path)
            safe_repo_path(Path(context.working_directory), input.path)
            if input.glob is not None and (
                input.glob.startswith("-") or ".." in Path(input.glob.replace("\\", "/")).parts
            ):
                raise ValueError("glob must not contain options or parent-directory segments")
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        return ValidationSuccess()

    async def call(self, input: GrepInput, context: ToolUseContext) -> ToolResult:
        root = Path(context.working_directory)
        state = self.instructions.activate_for_path(
            root, input.path, context.instruction_state
        )
        result = await asyncio.to_thread(self._search, root, input)
        if isinstance(result, ToolError):
            return ToolResult(error=result, context_update=ContextUpdate(instruction_state=state))
        return ToolResult(
            data=result,
            context_update=ContextUpdate(instruction_state=state),
        )

    def _search(self, root: Path, input: GrepInput) -> dict[str, object] | ToolError:
        arguments = [self.executable, "--json", "--color", "never"]
        if not input.case_sensitive:
            arguments.append("--ignore-case")
        if input.context_lines:
            arguments.extend(["--context", str(input.context_lines)])
        if input.glob is not None:
            arguments.extend(["--glob", input.glob])
        local_ignore = root / ".gitignore"
        if local_ignore.is_file():
            arguments.extend(["--ignore-file", str(local_ignore)])
        arguments.extend(["--", input.pattern, input.path])
        try:
            completed = subprocess.run(
                arguments,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                shell=False,
                env=build_subprocess_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolError(code="GREP_FAILED", message=str(exc), retryable=True)
        if completed.returncode not in {0, 1}:
            return ToolError(
                code="GREP_FAILED",
                message=(completed.stderr or "ripgrep failed").strip(),
                retryable=False,
            )
        matches: list[dict[str, object]] = []
        truncated = False
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in {"match", "context"}:
                continue
            data = event.get("data") or {}
            path = str((data.get("path") or {}).get("text") or "")
            line_number = data.get("line_number")
            text = str((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
            if not path or not isinstance(line_number, int):
                continue
            if len(matches) >= input.max_matches:
                truncated = True
                break
            matches.append(
                {
                    "path": Path(path).as_posix(),
                    "line": line_number,
                    "text": text,
                }
            )
        return {"matches": matches, "truncated": truncated}
