"""在受限 Docker 容器中执行仓库命令。"""

from __future__ import annotations

import asyncio
import os
import logging
from pathlib import Path, PurePosixPath
import shutil
import sys
from time import monotonic
from typing import Callable

from osc_agent.bot.models import RepositoryBotConfig
from osc_agent.runtime.models import ToolUseContext
from osc_agent.processes.contracts import CommandResult, ProcessRequest, ProcessRunner
from osc_agent.processes.policy import build_subprocess_environment
from osc_agent.workspaces.git_state import git_snapshot, git_workspace_fingerprint
from osc_agent.bot.policy import _matches
from osc_agent.bot.observability import log_event


class DockerProcessRunner(ProcessRunner):
    """每条仓库命令使用独立、无网络、只挂载 workspace 的容器。"""

    def __init__(
        self,
        *,
        workspace_root: Path,
        image_id: str,
        repository_config: RepositoryBotConfig,
        job_id: str,
        cancel_check: Callable[[], bool] | None = None,
        policy_violation_callback: Callable[[str], None] | None = None,
        docker_executable: str | None = None,
    ) -> None:
        if not image_id.startswith("sha256:"):
            raise ValueError("Docker sandbox requires an immutable sha256 image id")
        if sys.platform != "linux":
            raise ValueError("Docker bot sandbox is supported only on Linux hosts")
        self.workspace_root = workspace_root.resolve()
        self.image_id = image_id
        self.config = repository_config
        self.job_id = job_id
        self.cancel_check = cancel_check
        self.policy_violation_callback = policy_violation_callback
        uid = getattr(os, "getuid", lambda: 65_532)()
        gid = getattr(os, "getgid", lambda: 65_532)()
        if sys.platform == "linux" and uid == 0:
            raise ValueError("Docker bot worker must run as a non-root OS user")
        self.container_user = f"{uid}:{gid}"
        self.docker = docker_executable or shutil.which("docker") or ""
        if not self.docker:
            raise ValueError("Docker executable was not found; Host fallback is forbidden")

    async def run(
        self,
        request: ProcessRequest,
        context: ToolUseContext | None,
    ) -> CommandResult:
        repository = Path(request.repo_root).resolve()
        if not repository.is_relative_to(self.workspace_root):
            raise ValueError("Docker workspace is outside the configured bot workspace root")
        git_dir = repository / ".git"
        if not git_dir.is_dir() or git_dir.is_symlink():
            raise ValueError("Docker bot workspace must be a fresh clone with a real .git directory")
        if "," in str(repository):
            raise ValueError("Docker bind mount path cannot contain a comma")
        tool_id = context.tool_use_id if context is not None else "process"
        fingerprint_before = await asyncio.to_thread(
            git_workspace_fingerprint, repo_root=repository
        )
        name = _container_name(self.job_id, tool_id or "process")
        started = monotonic()
        process = await asyncio.create_subprocess_exec(
            *self._argv(name, repository, request.command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=build_subprocess_environment(),
        )
        termination_reason: str | None = None
        communicate = asyncio.create_task(process.communicate())
        cancel_monitor = asyncio.create_task(self._wait_for_cancel())
        try:
            done, _pending = await asyncio.wait(
                {communicate, cancel_monitor},
                timeout=min(
                    request.timeout_seconds,
                    float(self.config.command_timeout_seconds),
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                termination_reason = "timeout"
                await self._force_remove(name)
                stdout_bytes, stderr_bytes = await communicate
            elif cancel_monitor in done:
                try:
                    cancelled = cancel_monitor.result()
                except Exception:
                    await self._force_remove(name)
                    raise
                if cancelled:
                    termination_reason = "cancelled"
                    await self._force_remove(name)
                stdout_bytes, stderr_bytes = await communicate
            else:
                stdout_bytes, stderr_bytes = communicate.result()
        except asyncio.CancelledError:
            await self._force_remove(name)
            raise
        finally:
            cancel_monitor.cancel()
            await asyncio.gather(cancel_monitor, return_exceptions=True)
            if not communicate.done():
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await asyncio.gather(communicate, return_exceptions=True)
        result = CommandResult(
            command=request.command,
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_ms=max(0, int((monotonic() - started) * 1000)),
            termination_reason=termination_reason,
        )
        snapshot = await asyncio.to_thread(git_snapshot, repo_root=repository)
        fingerprint_after = await asyncio.to_thread(
            git_workspace_fingerprint, repo_root=repository
        )
        log_event(
            logging.getLogger("osc_agent.bot.process"),
            "process_git_fingerprint",
            service="worker",
            job_id=self.job_id,
            tool_use_id=tool_id,
            fingerprint_before=fingerprint_before,
            fingerprint_after=fingerprint_after,
            duration_ms=result.duration_ms,
        )
        files = [PurePosixPath(str(item["path"]).replace("\\", "/")) for item in snapshot["files"]]
        violation = next(
            (path.as_posix() for path in files if any(_matches(path, pattern) for pattern in self.config.denied_paths)),
            None,
        )
        if violation is not None:
            detail = f"repository process modified protected path: {violation}"
            if self.policy_violation_callback is not None:
                self.policy_violation_callback(detail)
            return result.model_copy(update={
                "exit_code": -5,
                "stderr": result.stderr + f"\n{detail}",
                "termination_reason": "repository_policy",
            })
        if len(files) > self.config.max_changed_files or len(str(snapshot["patch"]).encode()) > self.config.max_patch_bytes:
            detail = "repository process exceeded the configured change limits"
            if self.policy_violation_callback is not None:
                self.policy_violation_callback(detail)
            return result.model_copy(update={
                "exit_code": -5,
                "stderr": result.stderr + f"\n{detail}",
                "termination_reason": "repository_policy",
            })
        return result

    async def _wait_for_cancel(self) -> bool:
        if self.cancel_check is None:
            await asyncio.Future()
            return False
        while True:
            if await asyncio.to_thread(self.cancel_check):
                return True
            await asyncio.sleep(0.5)

    def _argv(self, name: str, repository: Path, command: str) -> list[str]:
        return [
            self.docker, "run", "--rm", "--name", name,
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(self.config.container_pids),
            "--memory", self.config.container_memory,
            "--cpus", str(self.config.container_cpus),
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev",
            "--tmpfs", "/home/osa:rw,nosuid,nodev",
            "--mount", f"type=bind,src={repository},dst=/workspace",
            "--mount", f"type=bind,src={repository / '.git'},dst=/workspace/.git,readonly",
            "--workdir", "/workspace", "--user", self.container_user,
            "--env", "CI=true", "--env", "NO_COLOR=1", "--env", "HOME=/home/osa",
            self.image_id,
            "/bin/bash", "--noprofile", "--norc", "-c", command,
        ]

    async def _force_remove(self, name: str) -> None:
        killer = await asyncio.create_subprocess_exec(
            self.docker, "rm", "-f", name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=build_subprocess_environment(),
        )
        _stdout, stderr = await killer.communicate()
        detail = stderr.decode("utf-8", errors="replace")[:500]
        missing = killer.returncode == 1 and "no such container" in detail.casefold()
        if killer.returncode != 0 and not missing:
            raise RuntimeError("failed to terminate Docker sandbox: " + stderr.decode("utf-8", errors="replace")[:500])


async def resolve_image_id(image: str, *, docker_executable: str | None = None) -> str:
    docker = docker_executable or shutil.which("docker") or ""
    if not docker:
        raise ValueError("Docker executable was not found")
    process = await asyncio.create_subprocess_exec(
        docker, "image", "inspect", "--format", "{{.Id}}", image,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=build_subprocess_environment(),
    )
    stdout, _stderr = await process.communicate()
    image_id = stdout.decode().strip()
    if process.returncode != 0 or not image_id.startswith("sha256:"):
        raise ValueError("configured Docker image is unavailable or has no immutable image id")
    return image_id


def _container_name(job_id: str, tool_use_id: str) -> str:
    safe_job = "".join(character for character in job_id.lower() if character.isalnum())[:12]
    safe_tool = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in tool_use_id.lower()
    )[:32]
    return f"osa-{safe_job}-{safe_tool}"[:63].rstrip("-")
