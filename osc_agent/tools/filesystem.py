"""将文件系统操作封装为带校验、权限和上下文更新的 Runtime 工具。"""

from __future__ import annotations

import asyncio
from hashlib import sha256
from pathlib import Path

from pydantic import Field

from osc_agent.workspaces.path_policy import (
    normalize_repo_relative_path,
    normalize_repo_relative_pattern,
)
from osc_agent.workspaces.path_policy import safe_repo_path
from osc_agent.contracts import ContractModel
from osc_agent.runtime.messages import ToolResultBlock, ToolUseBlock
from osc_agent.runtime.state import FilesObserved, InstructionsActivated, ToolContext
from osc_agent.runtime.tool_models import (
    ToolError,
    ToolResult,
    ValidationFailure,
    ValidationResult,
    ValidationSuccess,
)
from osc_agent.runtime.tool import BaseTool
from osc_agent.workspaces.files import edit_file, glob_files, write_file
from osc_agent.workspaces.instructions import RepositoryInstructionResolver
from osc_agent.workspaces.models import FileObservation


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
        context: ToolContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: ReadFileInput, context: ToolContext) -> ToolResult:
        root = Path(context.workspace.working_directory)
        try:
            target = safe_repo_path(root, input.path)
            if await asyncio.to_thread(_is_exact_unchanged_read, target, input, context):
                return ToolResult(
                    data={
                        "path": input.path,
                        "content": (
                            "File unchanged since last exact read. Refer to the earlier "
                            "read_file result instead of reading this range again."
                        ),
                        "offset": input.offset,
                        "complete": False,
                    }
                )
            text = await asyncio.to_thread(target.read_text, encoding="utf-8")
            stat = await asyncio.to_thread(target.stat)
            instruction_state = self.instructions.activate_for_path(
                root, input.path, context.workspace.instruction_state
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
            state_changes=(
                InstructionsActivated(state=instruction_state),
                FilesObserved(observations={input.path: observation}),
            ),
        )


def _is_exact_unchanged_read(
    target: Path,
    input: ReadFileInput,
    context: ToolContext,
) -> bool:
    observation = context.workspace.file_observations.get(input.path)
    if observation is None:
        return False
    try:
        stat = target.stat()
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if (
        stat.st_mtime_ns != observation.mtime_ns
        or sha256(text.encode("utf-8")).hexdigest() != observation.content_hash
    ):
        return False

    calls: dict[str, ReadFileInput] = {}
    exact_read_seen = False
    for message in context.transcript_messages:
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                if (
                    block.name in {"write_file", "edit_file"}
                    and block.input.get("path") == input.path
                ):
                    exact_read_seen = False
                if block.name == "read_file":
                    try:
                        calls[block.id] = ReadFileInput.model_validate(block.input)
                    except ValueError:
                        continue
            elif isinstance(block, ToolResultBlock) and not block.is_error:
                previous = calls.get(block.tool_use_id)
                if (
                    previous == input
                    and isinstance(block.content, dict)
                    and not block.content.get("error")
                ):
                    exact_read_seen = True
    return exact_read_seen


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

    def requires_approval(self, input: WriteFileInput) -> bool:
        # 创建或覆盖文件都必须经过统一 Permission pipeline。
        return True

    def permission_risk(self, input: WriteFileInput) -> str:
        return "write"

    def permission_preview(
        self,
        input: WriteFileInput,
        context: ToolContext,
    ) -> dict[str, object]:
        try:
            target = safe_repo_path(Path(context.workspace.working_directory), input.path)
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
        context: ToolContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: WriteFileInput, context: ToolContext) -> ToolResult:
        root = Path(context.workspace.working_directory)
        instruction_state = self.instructions.activate_for_path(
            root, input.path, context.workspace.instruction_state
        )
        if instruction_state != context.workspace.instruction_state:
            return ToolResult(
                error=ToolError(
                    code="REPOSITORY_INSTRUCTIONS_DISCOVERED",
                    message="new repository instructions were activated; review them and retry the write",
                    retryable=True,
                ),
                state_changes=(InstructionsActivated(state=instruction_state),),
            )
        stale = await asyncio.to_thread(_existing_file_guard, root, input.path, context)
        if stale is not None:
            return stale
        try:
            chars_written = await asyncio.to_thread(
                write_file,
                repo_root=root,
                path=input.path,
                content=input.content,
            )
            observation = await asyncio.to_thread(_observe_written_file, root, input.path)
        except (OSError, ValueError) as exc:
            return _file_error(exc)
        return ToolResult(
            data={"path": input.path, "chars_written": chars_written},
            state_changes=(FilesObserved(observations={input.path: observation}),),
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

    def requires_approval(self, input: EditFileInput) -> bool:
        return True

    def permission_risk(self, input: EditFileInput) -> str:
        return "write"

    def permission_preview(
        self,
        input: EditFileInput,
        context: ToolContext,
    ) -> dict[str, object]:
        return {
            "path": input.path,
            "old_text_chars": len(input.old_text),
            "new_text_chars": len(input.new_text),
        }

    async def validate_input(
        self,
        input: EditFileInput,
        context: ToolContext,
    ) -> ValidationResult:
        return _validate_path(input.path)

    async def call(self, input: EditFileInput, context: ToolContext) -> ToolResult:
        root = Path(context.workspace.working_directory)
        instruction_state = self.instructions.activate_for_path(
            root, input.path, context.workspace.instruction_state
        )
        if instruction_state != context.workspace.instruction_state:
            return ToolResult(
                error=ToolError(
                    code="REPOSITORY_INSTRUCTIONS_DISCOVERED",
                    message="new repository instructions were activated; review them and retry the edit",
                    retryable=True,
                ),
                state_changes=(InstructionsActivated(state=instruction_state),),
            )
        stale = await asyncio.to_thread(_existing_file_guard, root, input.path, context)
        if stale is not None:
            return stale
        try:
            replacements = await asyncio.to_thread(
                edit_file,
                repo_root=root,
                path=input.path,
                old_text=input.old_text,
                new_text=input.new_text,
            )
            observation = await asyncio.to_thread(_observe_written_file, root, input.path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return _file_error(exc)
        return ToolResult(
            data={"path": input.path, "replacements": replacements},
            state_changes=(FilesObserved(observations={input.path: observation}),),
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
        context: ToolContext,
    ) -> ValidationResult:
        try:
            normalize_repo_relative_pattern(input.pattern)
        except ValueError as exc:
            return ValidationFailure(reason=str(exc))
        return ValidationSuccess()

    async def call(self, input: GlobInput, context: ToolContext) -> ToolResult:
        try:
            paths = await asyncio.to_thread(
                glob_files,
                repo_root=Path(context.workspace.working_directory),
                pattern=input.pattern,
            )
        except (OSError, ValueError) as exc:
            return _file_error(exc)
        return ToolResult(data={"paths": paths})


def _validate_path(path: str) -> ValidationResult:
    try:
        normalize_repo_relative_path(path)
    except ValueError as exc:
        return ValidationFailure(reason=str(exc))
    return ValidationSuccess()


def _file_error(error: Exception) -> ToolResult:
    return ToolResult(error=ToolError(code="FILE_OPERATION_FAILED", message=str(error)))


def _existing_file_guard(
    root: Path,
    path: str,
    context: ToolContext,
) -> ToolResult | None:
    target = safe_repo_path(root, path)
    if not target.exists():
        return None
    observation = context.workspace.file_observations.get(path)
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
