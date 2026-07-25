from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path

from pydantic import Field

from osc_agent.tools.path_policy import (
    normalize_repo_relative_path,
    normalize_repo_relative_pattern,
)
from osc_agent.tools.path_policy import safe_repo_path
from osc_agent.runtime.instructions import RepositoryInstructionResolver
from osc_agent.runtime.models import (
    ContractModel,
    ContextUpdate,
    FileObservation,
    ToolError,
    ToolResult,
    ToolUseContext,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.tools.files import edit_file, glob_files, write_file


class ReadFileInput(ContractModel):
    path: str = Field(min_length=1)
    limit: int = Field(default=20_000, ge=1, le=200_000)
    offset: int = Field(default=0, ge=0)


class ReadFileOutput(ContractModel):
    path: str
    content: str
    offset: int = Field(ge=0)
    complete: bool


class ReadFileTool(BaseTool[ReadFileInput, ReadFileOutput]):
    name = "read_file"
    description = "Read a UTF-8 text file inside the current repository boundary."
    input_model = ReadFileInput
    output_model = ReadFileOutput

    def __init__(self, instructions: RepositoryInstructionResolver | None = None) -> None:
        self.instructions = instructions or RepositoryInstructionResolver()

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
        root = Path(context.working_directory)
        try:
            target = safe_repo_path(root, input.path)
            text = await asyncio.to_thread(target.read_text, encoding="utf-8")
            stat = await asyncio.to_thread(target.stat)
            instruction_state = self.instructions.activate_for_path(
                root, input.path, context.instruction_state
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return ToolResult(error=ToolError(code="FILE_OPERATION_FAILED", message=str(exc)))
        content = text[input.offset : input.offset + input.limit]
        complete = input.offset == 0 and len(content) == len(text)
        observation = FileObservation(
            path=input.path,
            content_hash=sha256(text.encode("utf-8")).hexdigest(),
            mtime_ns=stat.st_mtime_ns,
            complete=complete,
        )
        return ToolResult(
            data={
                "path": input.path,
                "content": content,
                "offset": input.offset,
                "complete": complete,
            },
            context_update=ContextUpdate(
                instruction_state=instruction_state,
                file_observations={input.path: observation},
            ),
        )


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

    def __init__(self, instructions: RepositoryInstructionResolver | None = None) -> None:
        self.instructions = instructions or RepositoryInstructionResolver()

    def is_destructive(self, input: WriteFileInput) -> bool:
        # 创建或覆盖文件都必须经过统一 Permission pipeline。
        return True

    def permission_risk(self, input: WriteFileInput) -> str:
        return "write"

    def permission_preview(
        self,
        input: WriteFileInput,
        context: ToolUseContext,
    ) -> dict[str, object]:
        try:
            target = safe_repo_path(Path(context.working_directory), input.path)
            operation = "replace" if target.exists() else "create"
        except ValueError:
            operation = "invalid"
        return {
            "path": input.path,
            "chars": len(input.content),
            "operation": operation,
        }

    async def validate_input(
        self,
        input: WriteFileInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: WriteFileInput, context: ToolUseContext) -> ToolResult:
        root = Path(context.working_directory)
        instruction_state = self.instructions.activate_for_path(
            root, input.path, context.instruction_state
        )
        if instruction_state != context.instruction_state:
            return ToolResult(
                error=ToolError(
                    code="REPOSITORY_INSTRUCTIONS_DISCOVERED",
                    message="new repository instructions were activated; review them and retry the write",
                    retryable=True,
                ),
                context_update=ContextUpdate(instruction_state=instruction_state),
            )
        stale = await asyncio.to_thread(_existing_file_guard, root, input.path, context)
        if stale is not None:
            return stale
        output = await asyncio.to_thread(
            write_file,
            repo_root=root,
            path=input.path,
            content=input.content,
            # 风险授权已由 ToolExecutor 统一完成，底层函数只负责安全写入算法。
            enforce_risk_checks=False,
        )
        if output.startswith("Error: "):
            return _file_error(output)
        observation = await asyncio.to_thread(_observe_written_file, root, input.path)
        return ToolResult(
            data={"path": input.path, "chars_written": len(input.content)},
            context_update=ContextUpdate(file_observations={input.path: observation}),
        )


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

    def __init__(self, instructions: RepositoryInstructionResolver | None = None) -> None:
        self.instructions = instructions or RepositoryInstructionResolver()

    def is_destructive(self, input: EditFileInput) -> bool:
        return True

    def permission_risk(self, input: EditFileInput) -> str:
        return "write"

    def permission_preview(
        self,
        input: EditFileInput,
        context: ToolUseContext,
    ) -> dict[str, object]:
        return {
            "path": input.path,
            "old_text_chars": len(input.old_text),
            "new_text_chars": len(input.new_text),
        }

    async def validate_input(
        self,
        input: EditFileInput,
        context: ToolUseContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: EditFileInput, context: ToolUseContext) -> ToolResult:
        root = Path(context.working_directory)
        instruction_state = self.instructions.activate_for_path(
            root, input.path, context.instruction_state
        )
        if instruction_state != context.instruction_state:
            return ToolResult(
                error=ToolError(
                    code="REPOSITORY_INSTRUCTIONS_DISCOVERED",
                    message="new repository instructions were activated; review them and retry the edit",
                    retryable=True,
                ),
                context_update=ContextUpdate(instruction_state=instruction_state),
            )
        stale = await asyncio.to_thread(_existing_file_guard, root, input.path, context)
        if stale is not None:
            return stale
        output = await asyncio.to_thread(
            edit_file,
            repo_root=root,
            path=input.path,
            old_text=input.old_text,
            new_text=input.new_text,
            enforce_risk_checks=False,
        )
        if output.startswith("Error: "):
            return _file_error(output)
        observation = await asyncio.to_thread(_observe_written_file, root, input.path)
        return ToolResult(
            data={"path": input.path, "replacements": 1},
            context_update=ContextUpdate(file_observations={input.path: observation}),
        )


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


def _existing_file_guard(
    root: Path,
    path: str,
    context: ToolUseContext,
) -> ToolResult | None:
    target = safe_repo_path(root, path)
    if not target.exists():
        return None
    observation = context.file_observations.get(path)
    if observation is None or not observation.complete:
        return ToolResult(
            error=ToolError(
                code="FILE_NOT_FULLY_READ",
                message="existing files must be read completely before editing or overwriting",
                retryable=True,
            )
        )
    try:
        text = target.read_text(encoding="utf-8")
        stat = target.stat()
    except (OSError, UnicodeDecodeError) as exc:
        return ToolResult(error=ToolError(code="FILE_OPERATION_FAILED", message=str(exc)))
    current_hash = sha256(text.encode("utf-8")).hexdigest()
    if current_hash != observation.content_hash or stat.st_mtime_ns != observation.mtime_ns:
        return ToolResult(
            error=ToolError(
                code="FILE_CHANGED_SINCE_READ",
                message="file changed after it was read; read the full file again before writing",
                retryable=True,
            )
        )
    return None


def _observe_written_file(root: Path, path: str) -> FileObservation:
    target = safe_repo_path(root, path)
    text = target.read_text(encoding="utf-8")
    return FileObservation(
        path=path,
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
        mtime_ns=target.stat().st_mtime_ns,
        complete=True,
    )
