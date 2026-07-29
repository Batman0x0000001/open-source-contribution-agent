"""实现宿主进程执行、禁用执行和进程树清理。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from osc_agent.processes.contracts import CommandResult, ProcessRequest
from osc_agent.runtime.models import ToolUseContext


class DisabledProcessRunner:
    """为禁止执行仓库进程的 Agent profile 提供 fail-closed runner。"""

    async def run(
        self,
        request: ProcessRequest,
        context: ToolUseContext | None,
    ) -> CommandResult:
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
    ) -> CommandResult:
        del context
        return await run_command(
            request.executable,
            request.command,
            repo_root=Path(request.repo_root),
            timeout_seconds=request.timeout_seconds,
            environment=request.environment,
        )


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
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
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
        return CommandResult(
            command=command,
            exit_code=-3,
            stderr=str(exc),
            duration_ms=int((time.perf_counter() - started) * 1000),
            termination_reason="os_error",
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
                    "/PID", str(process.pid), "/T", "/F",
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

