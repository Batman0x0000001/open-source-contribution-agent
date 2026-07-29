"""验证共享 Git 状态能力不依赖模型 Tool 适配层。"""

from __future__ import annotations

from pathlib import Path
import subprocess

from osc_agent.workspaces.git_state import git_snapshot, git_status, git_workspace_fingerprint


def _initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    (root / "tracked.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)


def test_git_state_reports_snapshot_and_stable_fingerprint(tmp_path: Path) -> None:
    _initialize_repository(tmp_path)
    baseline = git_workspace_fingerprint(repo_root=tmp_path)
    assert git_status(repo_root=tmp_path) == "(no output)"

    (tmp_path / "tracked.txt").write_text("after\n", encoding="utf-8")
    snapshot = git_snapshot(repo_root=tmp_path)

    assert snapshot["has_changes"] is True
    assert snapshot["files"][0]["path"] == "tracked.txt"
    assert git_workspace_fingerprint(repo_root=tmp_path) != baseline

