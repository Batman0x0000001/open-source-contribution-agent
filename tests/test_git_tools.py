from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from osc_agent.tools import git as git_tools


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    marker = path / ".gitkeep"
    marker.write_text("", encoding="utf-8")
    subprocess.run(["git", "add", ".gitkeep"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_git_failures_are_not_parsed_as_repository_state(tmp_path):
    assert git_tools.git_status(repo_root=tmp_path).startswith("Error:")
    assert git_tools.git_changed_files(repo_root=tmp_path) == []
    assert git_tools.git_diff_numstat(repo_root=tmp_path) == (0, 0)


def test_repository_without_head_is_not_supported(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    with pytest.raises(RuntimeError, match="git command failed"):
        git_tools.git_diff_numstat(repo_root=tmp_path)


def test_untracked_directory_is_expanded_to_files_and_counted(tmp_path):
    _init_repo(tmp_path)
    source = tmp_path / "newdir" / "large.py"
    source.parent.mkdir()
    source.write_text("x = 1\n" * 200, encoding="utf-8")
    metadata = tmp_path / ".osc_agent" / "trace.jsonl"
    metadata.parent.mkdir()
    metadata.write_text("internal\n" * 500, encoding="utf-8")

    assert git_tools.git_changed_files(repo_root=tmp_path) == [".osc_agent/trace.jsonl", "newdir/large.py"]
    assert git_tools.git_diff_numstat(repo_root=tmp_path) == (200, 0)


def test_git_diff_disables_external_diff_helpers(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    git_tools.git_diff(repo_root=tmp_path)

    assert "--no-ext-diff" in captured["args"]
    assert "--no-textconv" in captured["args"]
    assert captured["env"]["GIT_OPTIONAL_LOCKS"] == "0"


def test_machine_readable_status_is_not_truncated(tmp_path, monkeypatch):
    records = "".join(f"?? generated/file-{index:05d}.py\0" for index in range(3_000))

    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=records, stderr="")

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    files = git_tools.git_changed_files(repo_root=tmp_path)

    assert len(files) == 3_000
    assert files[-1] == "generated/file-02999.py"
