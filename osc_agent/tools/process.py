from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
import re
import time

from pydantic import BaseModel, ConfigDict


class CommandKind(str, Enum):
    TEST = "test"
    BUILD = "build"
    LINT = "lint"
    TYPECHECK = "typecheck"
    OTHER = "other"


class CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int
    termination_reason: str | None = None

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()


def classify_command(command: str) -> CommandKind:
    lowered = command.casefold()
    test_patterns = (
        r"(^|\s)(pytest|py\.test|tox|nox)(\s|$)",
        r"python\s+-m\s+(pytest|unittest)",
        r"(^|\s)(npm|pnpm|yarn)\s+(run\s+)?test(\s|$)",
        r"(^|\s)cargo\s+test(\s|$)",
        r"(^|\s)go\s+test(\s|$)",
        r"(^|\s)dotnet\s+test(\s|$)",
        r"(^|\s)(mvn|mvnw|gradle|gradlew)(\.cmd)?\s+.*\btest\b",
    )
    if any(re.search(pattern, lowered) for pattern in test_patterns):
        return CommandKind.TEST
    if re.search(r"(^|\s)(npm|pnpm|yarn)\s+(run\s+)?build(\s|$)|(^|\s)(cargo|go|dotnet)\s+build(\s|$)", lowered):
        return CommandKind.BUILD
    if re.search(r"(^|\s)(ruff|eslint|pylint)(\s|$)|(^|\s)(npm|pnpm|yarn)\s+(run\s+)?lint(\s|$)", lowered):
        return CommandKind.LINT
    if re.search(r"(^|\s)(mypy|pyright|tsc)(\s|$)|(^|\s)(npm|pnpm|yarn)\s+(run\s+)?typecheck(\s|$)", lowered):
        return CommandKind.TYPECHECK
    return CommandKind.OTHER


async def run_command(
    executable: str,
    command: str,
    *,
    repo_root: Path,
    timeout_seconds: int | float,
) -> CommandResult:
    started = time.perf_counter()
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
            cwd=repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            return CommandResult(
                command=command,
                exit_code=-1,
                stderr=f"command exceeded {timeout_seconds} seconds",
                duration_ms=int((time.perf_counter() - started) * 1000),
                termination_reason="timeout",
            )
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        reason = None if process.returncode == 0 else "nonzero_exit"
        return CommandResult(
            command=command,
            exit_code=process.returncode or 0,
            stdout=stdout,
            stderr=stderr,
            duration_ms=int((time.perf_counter() - started) * 1000),
            termination_reason=reason,
        )
    except OSError as exc:
        return CommandResult(command=command, exit_code=-3, stderr=str(exc), duration_ms=int((time.perf_counter() - started) * 1000), termination_reason="os_error")
