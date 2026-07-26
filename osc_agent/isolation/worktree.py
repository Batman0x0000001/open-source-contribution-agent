from __future__ import annotations

import re
import subprocess
from pathlib import Path

from osc_agent.runtime.models import WorktreeSession
from osc_agent.tools.process import build_subprocess_environment


class WorktreeManager:
    """以真实 Git worktree 提供执行目录隔离，不维护额外任务状态。"""

    def __init__(self, worktrees_root: Path) -> None:
        self.worktrees_root = worktrees_root.resolve()

    def create(self, repo_root: Path, name: str) -> WorktreeSession:
        self.validate_name(name)
        root = self.repository_root(repo_root)
        target = (self.worktrees_root / name).resolve()
        if target.exists():
            raise ValueError(f"worktree already exists: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        branch = f"osc-agent/{name}"
        base_commit = self._git(root, "rev-parse", "HEAD").strip()
        self._git(root, "worktree", "add", str(target), "-b", branch, "HEAD")
        return WorktreeSession(
            path=str(target),
            original_working_directory=str(root),
            branch=branch,
            base_commit=base_commit,
        )

    def is_dirty(self, session: WorktreeSession) -> bool:
        output = self._git(Path(session.path), "status", "--porcelain")
        return bool(output.strip())

    def commits_ahead(self, session: WorktreeSession) -> int:
        output = self._git(Path(session.path), "rev-list", "--count", f"{session.base_commit}..HEAD")
        try:
            return int(output.strip())
        except ValueError as exc:
            raise ValueError("git returned an invalid commit count") from exc

    def validate_session(self, session: WorktreeSession) -> None:
        target = Path(session.path).resolve()
        root = Path(session.original_working_directory).resolve()
        if not target.is_dir():
            raise ValueError("saved worktree directory no longer exists")
        registered = self._git(root, "worktree", "list", "--porcelain")
        entries = {
            Path(line.removeprefix("worktree ")).resolve()
            for line in registered.splitlines()
            if line.startswith("worktree ")
        }
        if target not in entries:
            raise ValueError("saved worktree is no longer registered by git")
        if self._common_git_directory(target) != self._common_git_directory(root):
            raise ValueError("saved worktree belongs to a different repository")

    def remove(self, session: WorktreeSession, *, discard: bool = False) -> None:
        target = Path(session.path).resolve()
        root = Path(session.original_working_directory).resolve()
        self.validate_session(session)
        if not discard:
            if self.is_dirty(session):
                raise ValueError("worktree has uncommitted changes; keep it or explicitly discard all work")
            if self.commits_ahead(session) > 0:
                raise ValueError("worktree has commits beyond its base; keep it or explicitly discard all work")
        arguments = ["worktree", "remove", str(target)]
        if discard:
            arguments.append("--force")
        self._git(root, *arguments)
        self._git(root, "branch", "-D" if discard else "-d", session.branch)

    @staticmethod
    def validate_name(name: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name) or name in {".", ".."}:
            raise ValueError("worktree name must match [A-Za-z0-9._-]{1,64}")

    @classmethod
    def repository_root(cls, path: Path) -> Path:
        value = cls._git(path.resolve(), "rev-parse", "--show-toplevel")
        return Path(value.strip()).resolve()

    @classmethod
    def _common_git_directory(cls, path: Path) -> Path:
        value = cls._git(path.resolve(), "rev-parse", "--path-format=absolute", "--git-common-dir")
        return Path(value.strip()).resolve()

    @staticmethod
    def _git(cwd: Path, *arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={cwd.resolve()}", *arguments],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                env=build_subprocess_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"git command failed: {exc}") from exc
        if completed.returncode != 0:
            output = ((completed.stdout or "") + (completed.stderr or "")).strip()
            raise ValueError(output or "git command failed")
        return (completed.stdout or "").strip()
