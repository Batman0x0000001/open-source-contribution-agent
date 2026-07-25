from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from osc_agent.isolation.worktree import WorktreeManager
from osc_agent.runtime.models import WorktreeSession


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_real_worktree_create_dirty_protection_and_cleanup(tmp_path: Path) -> None:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "tests@example.com")
    git(tmp_path, "config", "user.name", "Tests")
    (tmp_path / "tracked.txt").write_text("base", encoding="utf-8")
    git(tmp_path, "add", "tracked.txt")
    git(tmp_path, "commit", "-m", "base")
    manager = WorktreeManager(tmp_path.parent / f"{tmp_path.name}-worktrees")

    session = manager.create(tmp_path, "feature")
    target = Path(session.path)
    assert target.is_dir()
    assert session.branch == "osc-agent/feature"

    (target / "tracked.txt").write_text("changed", encoding="utf-8")
    assert manager.is_dirty(session)
    with pytest.raises(ValueError, match="uncommitted changes"):
        manager.remove(session)

    manager.remove(session, discard=True)
    assert not target.exists()


def test_committed_work_is_not_removed_without_explicit_discard(tmp_path: Path) -> None:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "tests@example.com")
    git(tmp_path, "config", "user.name", "Tests")
    (tmp_path / "tracked.txt").write_text("base", encoding="utf-8")
    git(tmp_path, "add", "tracked.txt")
    git(tmp_path, "commit", "-m", "base")
    manager = WorktreeManager(tmp_path.parent / f"{tmp_path.name}-worktrees")
    session = manager.create(tmp_path, "committed")
    target = Path(session.path)
    (target / "tracked.txt").write_text("committed change", encoding="utf-8")
    git(target, "add", "tracked.txt")
    git(target, "commit", "-m", "work")

    assert manager.commits_ahead(session) == 1
    with pytest.raises(ValueError, match="commits beyond its base"):
        manager.remove(session)
    assert target.exists()

    manager.remove(session, discard=True)
    assert not target.exists()


def test_git_inspection_failure_keeps_worktree(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "worktree"
    target.mkdir()
    manager = WorktreeManager(tmp_path / "state")
    session = WorktreeSession(
        path=str(target),
        original_working_directory=str(tmp_path),
        branch="osc-agent/fail-closed",
        base_commit="abc123",
    )
    monkeypatch.setattr(manager, "validate_session", lambda _session: None)

    def fail_inspection(_session) -> bool:
        raise ValueError("git failed")

    monkeypatch.setattr(manager, "is_dirty", fail_inspection)

    with pytest.raises(ValueError, match="git failed"):
        manager.remove(session)
    assert target.exists()
