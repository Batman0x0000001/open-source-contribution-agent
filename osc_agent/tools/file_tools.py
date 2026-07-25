from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import Field

from osc_agent.tools.path_policy import (
    normalize_repo_relative_path,
    normalize_repo_relative_pattern,
)
from osc_agent.runtime.models import (
    ContractModel,
    ToolError,
    ToolResult,
    ToolUseContext,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.tools.files import edit_file, glob_files, read_file, write_file


class ReadFileInput(ContractModel):
    path: str = Field(min_length=1)
    limit: int = Field(default=20_000, ge=1, le=200_000)
    offset: int = Field(default=0, ge=0)


class ReadFileOutput(ContractModel):
    path: str
    content: str
    offset: int = Field(ge=0)


class ReadFileTool(BaseTool[ReadFileInput, ReadFileOutput]):
    name = "read_file"
    description = "Read a UTF-8 text file inside the current repository boundary."
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def is_read_only(self, input: ReadFileInput) -> bool:
        return True

    def is_concurrency_safe(self, input: ReadFileInput) -> bool:
        return True

    async def validate_input(
        self,
        input: ReadFileInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: ReadFileInput, context: ToolUseContext) -> ToolResult:
        content = await asyncio.to_thread(
            read_file,
            repo_root=Path(context.working_directory),
            path=input.path,
            limit=input.limit,
            offset=input.offset,
        )
        if content.startswith("Error: "):
            return _file_error(content)
        return ToolResult(data={"path": input.path, "content": content, "offset": input.offset})


class WriteFileInput(ContractModel):
    path: str = Field(min_length=1)
    content: str


class WriteFileOutput(ContractModel):
    path: str
    chars_written: int = Field(ge=0)


class WriteFileTool(BaseTool[WriteFileInput, WriteFileOutput]):
    name = "write_file"
    description = "Create or replace a UTF-8 text file inside the current repository boundary."
    input_model = WriteFileInput
    output_model = WriteFileOutput

    def is_destructive(self, input: WriteFileInput) -> bool:
        # 创建或覆盖文件都必须经过统一 Permission pipeline。
        return True

    async def validate_input(
        self,
        input: WriteFileInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: WriteFileInput, context: ToolUseContext) -> ToolResult:
        output = await asyncio.to_thread(
            write_file,
            repo_root=Path(context.working_directory),
            path=input.path,
            content=input.content,
            # 风险授权已由 ToolExecutor 统一完成，底层函数只负责安全写入算法。
            enforce_risk_checks=False,
        )
        if output.startswith("Error: "):
            return _file_error(output)
        return ToolResult(data={"path": input.path, "chars_written": len(input.content)})


class EditFileInput(ContractModel):
    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str


class EditFileOutput(ContractModel):
    path: str
    replacements: int = Field(ge=1)


class EditFileTool(BaseTool[EditFileInput, EditFileOutput]):
    name = "edit_file"
    description = "Replace one exact text occurrence in a repository file."
    input_model = EditFileInput
    output_model = EditFileOutput

    def is_destructive(self, input: EditFileInput) -> bool:
        return True

    async def validate_input(
        self,
        input: EditFileInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: EditFileInput, context: ToolUseContext) -> ToolResult:
        output = await asyncio.to_thread(
            edit_file,
            repo_root=Path(context.working_directory),
            path=input.path,
            old_text=input.old_text,
            new_text=input.new_text,
            enforce_risk_checks=False,
        )
        if output.startswith("Error: "):
            return _file_error(output)
        return ToolResult(data={"path": input.path, "replacements": 1})


class GlobInput(ContractModel):
    pattern: str = Field(min_length=1)


class GlobOutput(ContractModel):
    paths: list[str]


class GlobTool(BaseTool[GlobInput, GlobOutput]):
    name = "glob"
    description = "List repository files matching a glob pattern without reading their contents."
    input_model = GlobInput
    output_model = GlobOutput

    def is_read_only(self, input: GlobInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GlobInput) -> bool:
        return True

    async def validate_input(
        self,
        input: GlobInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        try:
            normalize_repo_relative_pattern(input.pattern)
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        return ValidationSuccess()

    async def call(self, input: GlobInput, context: ToolUseContext) -> ToolResult:
        output = await asyncio.to_thread(
            glob_files,
            repo_root=Path(context.working_directory),
            pattern=input.pattern,
        )
        if output.startswith("Error: "):
            return _file_error(output)
        paths = [] if output == "(no matches)" else output.splitlines()
        return ToolResult(data={"paths": paths})


def _validate_path(path: str) -> ValidationResult:
    try:
        normalize_repo_relative_path(path)
    except ValueError as exc:
        return ValidationFailure(reason=str(exc))
    return ValidationSuccess()


def _file_error(output: str) -> ToolResult:
    return ToolResult(error=ToolError(code="FILE_OPERATION_FAILED", message=output.removeprefix("Error: ")))
