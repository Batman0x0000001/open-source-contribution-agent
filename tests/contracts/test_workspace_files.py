"""验证共享工作区文件能力的结构化返回与安全边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

from osc_agent.workspaces.files import edit_file, glob_files, write_file


def test_workspace_files_return_structured_values(tmp_path: Path) -> None:
    assert write_file(
        repo_root=tmp_path,
        path="src/example.py",
        content="old old\n",
    ) == 8
    assert edit_file(
        repo_root=tmp_path,
        path="src/example.py",
        old_text="old",
        new_text="new",
    ) == 1
    assert glob_files(repo_root=tmp_path, pattern="src/*.py") == ["src/example.py"]
    assert (tmp_path / "src" / "example.py").read_text(encoding="utf-8") == "new old\n"


def test_workspace_files_raise_instead_of_returning_error_strings(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="old_text not found"):
        edit_file(
            repo_root=tmp_path,
            path="example.txt",
            old_text="missing",
            new_text="new",
        )

    with pytest.raises(ValueError, match="escapes repository"):
        write_file(repo_root=tmp_path, path="../outside.txt", content="no")

    assert glob_files(repo_root=tmp_path, pattern="*.missing") == []
