from __future__ import annotations

import asyncio
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from osc_agent.runtime.models import FrozenContractModel, ToolUseContext


DEFAULT_SUBPROCESS_ENV_NAMES = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "TERM",
        "NO_COLOR",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "JAVA_HOME",
        "DOTNET_ROOT",
    }
)
_PROTECTED_ENV_EXACT = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "RIPGREP_CONFIG_PATH",
        "SSH_AUTH_SOCK",
    }
)
_PROTECTED_ENV_MARKERS = (
    "API_KEY",
    "AUTH_TOKEN",
    "AUTHORIZATION",
    "BEARER_TOKEN",
    "CLIENT_SECRET",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION_TOKEN",
)


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


class ProcessRequest(FrozenContractModel):
    executable: str
    command: str
    repo_root: str
    timeout_seconds: float
    environment: dict[str, str]


class ProcessRunner(Protocol):
    async def run(
        self,
        request: ProcessRequest,
        context: ToolUseContext | None,
    ) -> "CommandResult": ...


class DisabledProcessRunner:
    """Fail closed for profiles that must never execute repository processes."""

    async def run(
        self,
        request: ProcessRequest,
        context: ToolUseContext | None,
    ) -> "CommandResult":
        del context
        return CommandResult(
            command=request.command,
            exit_code=-4,
            stderr="process execution is disabled for this Agent profile",
            duration_ms=0,
            termination_reason="disabled",
        )


class HostProcessRunner:
    async def run(
        self,
        request: ProcessRequest,
        context: ToolUseContext | None,
    ) -> "CommandResult":
        del context
        return await run_command(
            request.executable,
            request.command,
            repo_root=Path(request.repo_root),
            timeout_seconds=request.timeout_seconds,
            environment=request.environment,
        )

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
    environment: Mapping[str, str],
) -> CommandResult:
    started = time.perf_counter()
    try:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        shell_name = Path(executable).stem.casefold()
        shell_arguments = (
            ["--noprofile", "--norc", "-c", command]
            if shell_name == "bash"
            else ["-c", command]
        )
        process = await asyncio.create_subprocess_exec(
            executable,
            *shell_arguments,
            cwd=repo_root,
            env=dict(environment),
            creationflags=creationflags,
            start_new_session=os.name != "nt",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except asyncio.CancelledError:
            await _terminate_process_tree(process, environment)
            raise
        except TimeoutError:
            cleanup_error = await _terminate_process_tree(process, environment)
            detail = f"command exceeded {timeout_seconds} seconds"
            if cleanup_error:
                detail += f"; process-tree cleanup warning: {cleanup_error}"
            return CommandResult(
                command=command,
                exit_code=-1,
                stderr=detail,
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


def build_subprocess_environment(
    additional_names: Iterable[str] = (),
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """只复制运行必需和用户显式允许的非敏感环境变量。"""

    available = os.environ if source is None else source
    requested = {name.strip().upper() for name in additional_names if name.strip()}
    names = DEFAULT_SUBPROCESS_ENV_NAMES | requested
    environment: dict[str, str] = {}
    for name, value in available.items():
        canonical = name.upper()
        if canonical in names and not is_protected_environment_name(canonical):
            environment[name] = value
    return environment


def is_protected_environment_name(name: str) -> bool:
    canonical = name.strip().upper()
    return canonical in _PROTECTED_ENV_EXACT or any(
        marker in canonical for marker in _PROTECTED_ENV_MARKERS
    )


async def _terminate_process_tree(
    process: asyncio.subprocess.Process,
    environment: Mapping[str, str],
) -> str | None:
    if process.returncode is not None:
        return None
    cleanup_error: str | None = None
    if os.name == "nt":
        system_root = environment.get("SystemRoot") or environment.get("SYSTEMROOT")
        taskkill = (
            str(Path(system_root) / "System32" / "taskkill.exe")
            if system_root
            else shutil.which("taskkill", path=environment.get("PATH"))
        )
        if taskkill:
            try:
                killer = await asyncio.create_subprocess_exec(
                    taskkill,
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    env=dict(environment),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(killer.communicate(), timeout=10)
                if killer.returncode not in {0, 128}:
                    cleanup_error = (
                        stderr.decode("utf-8", errors="replace").strip()
                        or stdout.decode("utf-8", errors="replace").strip()
                        or f"taskkill exited with code {killer.returncode}"
                    )
            except (OSError, TimeoutError) as exc:
                cleanup_error = str(exc)
        else:
            cleanup_error = "taskkill.exe was not found"
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            cleanup_error = str(exc)
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        cleanup_error = cleanup_error or "process did not exit after termination"
    return cleanup_error
