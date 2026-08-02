"""定义工作区、仓库指令和文件观察状态。"""

from typing import Literal

from pydantic import Field

from osc_agent.contracts import FrozenContractModel


class WorktreeSession(FrozenContractModel):
    path: str = Field(min_length=1)
    original_working_directory: str = Field(min_length=1)
    branch: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)


class InstructionDocument(FrozenContractModel):
    path: str = Field(min_length=1)
    kind: Literal["agents", "claude"]
    scope_directory: str
    content_hash: str = Field(min_length=1)
    content: str = Field(min_length=1)


class RepositoryInstructionState(FrozenContractModel):
    active_paths: tuple[str, ...] = ()


class FileObservation(FrozenContractModel):
    path: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    mtime_ns: int = Field(ge=0)
    complete: bool
