from __future__ import annotations

from pathlib import Path

import pytest

from osc_agent.harness.permissions import check_shell_command, safe_repo_path
from osc_agent.tools.files import edit_file, read_file, write_file
from osc_agent.tools.shell import run_bash


def test_safe_repo_path_blocks_path_escape(tmp_path):
    with pytest.raises(ValueError):
        safe_repo_path(tmp_path, "../outside.txt")


def test_file_tools_reject_path_escape(tmp_path):
    assert "path escapes repository" in read_file(repo_root=tmp_path, path="../outside.txt")
    assert "path escapes repository" in write_file(
        repo_root=tmp_path,
        path="../outside.txt",
        content="nope",
    )


def test_edit_file_rejects_path_escape(tmp_path):
    assert "path escapes repository" in edit_file(
        repo_root=tmp_path,
        path="../outside.txt",
        old_text="a",
        new_text="b",
    )


def test_dangerous_shell_command_is_denied(tmp_path):
    output = run_bash("git push origin main", repo_root=tmp_path)

    assert output.startswith("Permission denied:")


def test_suspicious_shell_command_requires_confirmation():
    decision = check_shell_command("pip install some-package")

    assert decision.action == "ask"
    assert "explicit confirmation" in decision.reason


def test_large_file_write_requires_confirmation(tmp_path):
    output = write_file(repo_root=tmp_path, path="large.txt", content="x" * 500_001)

    assert output.startswith("Permission required:")
