from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
import hashlib
import re
import subprocess
import time

from pydantic import BaseModel, ConfigDict


class CommandKind(str, Enum):
    TEST = "test"
    OTHER = "other"


_COMMAND_SEPARATOR = re.compile(r"\s*(?:&&|\|\||[;|])\s*")
_RUN_WRAPPERS = {"uv", "poetry", "pipenv"}
_DIRECT_TEST_RUNNERS = {"pytest", "py.test", "tox", "nox"}
_TEST_SUBCOMMAND_RUNNERS = {"npm", "pnpm", "yarn", "bun", "cargo", "go", "dotnet"}


class CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int
    artifact_path: str = ""
    termination_reason: str | None = None

    @property
    def output(self) -> str:
        return ((self.stdout or "") + (self.stderr or "")).strip()


def classify_command(command: str) -> CommandKind:
    """按可执行命令结构识别测试调用，避免把 `echo pytest` 当成测试。"""
    for segment in _COMMAND_SEPARATOR.split(command.casefold()):
        tokens = segment.strip().split()
        while tokens and "=" in tokens[0] and not tokens[0].startswith(("-", "/")):
            tokens.pop(0)
        if len(tokens) >= 2 and _executable_name(tokens[0]) in _RUN_WRAPPERS and tokens[1] == "run":
            tokens = tokens[2:]
        if _is_test_tokens(tokens):
            return CommandKind.TEST
    return CommandKind.OTHER


def _is_test_tokens(tokens: list[str]) -> bool:
    if not tokens:
        return False
    executable = _executable_name(tokens[0])
    if executable in _DIRECT_TEST_RUNNERS:
        return True
    if executable in {"python", "python3", "py"} and len(tokens) >= 3 and tokens[1] == "-m":
        return tokens[2] in {"pytest", "unittest", "tox", "nox"}
    if executable in _TEST_SUBCOMMAND_RUNNERS and len(tokens) >= 2:
        return tokens[1] in {"test", "tests"}
    if executable in {"mvn", "mvnw", "gradle", "gradlew"}:
        return any(token == "test" or token.endswith(":test") for token in tokens[1:])
    return False


def _executable_name(value: str) -> str:
    name = Path(value.strip('"\'')).name.casefold()
    return name.removesuffix(".exe")


def run_command(
    command: str,
    *,
    repo_root: Path,
    timeout_seconds: int | float,
    artifact_namespace: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> CommandResult:
    """统一执行宿主机命令；权限判断必须由调用方在进入这里前完成。"""
    started = time.perf_counter()
    exit_code = 0
    stdout = ""
    stderr = ""
    termination_reason: str | None = None
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=dict(environment) if environment is not None else None,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if exit_code != 0:
            termination_reason = "nonzero_exit"
    except subprocess.TimeoutExpired as exc:
        exit_code = -1
        stderr = f"command timed out after {timeout_seconds} seconds: {exc}"
        termination_reason = "timeout"
    except OSError as exc:
        exit_code = -3
        stderr = str(exc)
        termination_reason = "os_error"

    artifact_path = ""
    if artifact_namespace:
        log_dir = repo_root / ".osc_agent" / artifact_namespace
        log_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(command.encode("utf-8")).hexdigest()[:12]
        path = log_dir / f"{digest}.log"
        path.write_text(((stdout or "") + (stderr or "")).strip() + "\n", encoding="utf-8")
        artifact_path = str(path)

    return CommandResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=int((time.perf_counter() - started) * 1000),
        artifact_path=artifact_path,
        termination_reason=termination_reason,
    )
