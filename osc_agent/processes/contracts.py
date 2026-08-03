"""定义受控进程执行的稳定请求与结果契约。"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from osc_agent.contracts import FrozenContractModel


class CommandKind(str, Enum):
    TEST = "test"
    BUILD = "build"
    LINT = "lint"
    TYPECHECK = "typecheck"
    OTHER = "other"


class CommandResult(FrozenContractModel):
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int
    termination_reason: str | None = None

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()


class ProcessRequest(FrozenContractModel):
    invocation_id: str
    executable: str
    command: str
    repo_root: str
    timeout_seconds: float
    environment: dict[str, str]


class ProcessRunner(Protocol):
    async def run(self, request: ProcessRequest) -> CommandResult: ...
