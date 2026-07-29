"""创建和验证 Bot 作业的独立 Git clone 工作区。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
from typing import Literal
from uuid import uuid4

from osc_agent.bot.github_app import GitHubControlClient, basic_git_auth_header
from osc_agent.bot.models import BotJob
from osc_agent.bot.store import BotStore
from osc_agent.processes.policy import build_subprocess_environment


class BotJobWorkspacePreparer:
    def __init__(self, *, root: Path, store: BotStore, github: GitHubControlClient) -> None:
        self.root = root.resolve()
        self.store = store
        self.github = github
        self.git = shutil.which("git") or ""
        if not self.git:
            raise ValueError("Git executable was not found")

    async def prepare(self, job: BotJob, phase: Literal["plan", "implementation"]) -> BotJob:
        target = (self.root / job.job_id / phase).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("bot workspace escapes the configured root")
        if target.exists():
            try:
                await self._validate_existing(target, job)
            except ValueError:
                shutil.rmtree(target)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.parent / f".{phase}.preparing-{uuid4().hex}"
            token = await self.github.installation_token(job.installation_id, contents="read")
            env = build_subprocess_environment()
            env.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
                    "GIT_CONFIG_VALUE_0": basic_git_auth_header(token),
                    "GIT_TERMINAL_PROMPT": "0",
                }
            )
            try:
                await self._git(
                    ["clone", "--no-checkout", f"https://github.com/{job.repository_full_name}.git", str(staging)],
                    cwd=self.root,
                    env=env,
                )
                await self._git(["checkout", "--detach", job.base_sha], cwd=staging)
                await self._git(["remote", "set-url", "origin", f"https://github.com/{job.repository_full_name}.git"], cwd=staging)
                staging.replace(target)
            except Exception:
                if staging.is_relative_to(self.root) and staging.exists():
                    shutil.rmtree(staging)
                raise
        current = self.store.get_job(job.job_id)
        if current is None:
            raise ValueError("bot job disappeared during workspace preparation")
        phase_changes = (
            {"plan_workspace_path": str(target), "plan_workspace_ready": True}
            if phase == "plan"
            else {"implementation_workspace_path": str(target), "implementation_workspace_ready": True}
        )
        return self.store.update_job_fields(
            job.job_id,
            expected_version=current.version,
            **phase_changes,
        )

    async def _validate_existing(self, target: Path, job: BotJob) -> None:
        head = (await self._git(["rev-parse", "HEAD"], cwd=target)).strip()
        origin = (await self._git(["remote", "get-url", "origin"], cwd=target)).strip()
        expected = f"https://github.com/{job.repository_full_name}.git"
        if head != job.base_sha or origin != expected or not (target / ".git").is_dir():
            raise ValueError("existing bot workspace does not match the queued job")

    async def _git(self, arguments: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
        process = await asyncio.create_subprocess_exec(
            self.git,
            *arguments,
            cwd=cwd,
            env=env or build_subprocess_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise ValueError("trusted Git workspace operation failed: " + stderr.decode(errors="replace")[:1_000])
        return stdout.decode("utf-8", errors="replace")
