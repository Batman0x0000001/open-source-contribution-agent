"""
模型请求 Git 工具
      ↓
调用 git_status / git_diff / git_log
      ↓
统一调用 _run_git()
      ↓
执行 git 命令
      ↓
返回输出给 Agent
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import subprocess
from pathlib import Path

from osc_agent.harness.repository_boundary import normalize_repo_relative_path, safe_repo_path

GIT_TOOLS = [
    {
        "name": "git_status",
        "description": "Show git status for the target repository.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_diff",
        "description": "Show the current git diff for the target repository.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_log",
        "description": "Show recent git commits for the target repository.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 5}},
        },
    },
]

MAX_GIT_OUTPUT_CHARS = 50_000


class GitChangeKind(str, Enum):
    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"
    RENAMED = "RENAMED"
    COPIED = "COPIED"
    UNMERGED = "UNMERGED"


@dataclass(frozen=True)
class GitChange:
    kind: GitChangeKind
    path: str
    old_path: str | None = None
    added_lines: int = 0
    deleted_lines: int = 0
    binary: bool = False


def _run_git(
    repo_root: Path,
    args: list[str],
    *,
    max_output_chars: int | None = MAX_GIT_OUTPUT_CHARS,
) -> str:
    """用参数列表调用 git，避免通过 shell 拼接只读 git 命令。"""
    try:
        completed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Error: {exc}"

    output = (completed.stdout or "").strip("\r\n")
    if completed.returncode != 0:
        detail = "\n".join(part for part in (output, (completed.stderr or "").strip("\r\n")) if part)
        return f"Error: git command failed with exit code {completed.returncode}: {detail}"[:MAX_GIT_OUTPUT_CHARS]
    result = output or "(no output)"
    return result if max_output_chars is None else result[:max_output_chars]


def git_status(*, repo_root: Path) -> str:
    """读取当前仓库状态，不执行任何写操作。"""
    return _run_git(repo_root, ["status", "--short"])


def git_diff(*, repo_root: Path) -> str:
    """读取当前工作区 diff，用于后续贡献摘要。"""
    return _run_git(repo_root, ["diff", "--no-ext-diff", "--no-textconv", "--"])


def git_log(*, repo_root: Path, limit: int = 5) -> str:
    """读取最近提交，limit 做最小夹取避免过长输出。"""
    safe_limit = min(max(int(limit), 1), 50)
    return _run_git(repo_root, ["log", f"-{safe_limit}", "--oneline"])


def git_head(*, repo_root: Path) -> str:
    """读取当前 HEAD；仓库无提交或命令失败时返回显式错误。"""
    return _run_git(repo_root, ["rev-parse", "HEAD"])


def git_common_dir(*, repo_root: Path) -> str:
    """返回所有关联 worktree 共享的 Git 元数据目录。"""
    return _run_git(repo_root, ["rev-parse", "--git-common-dir"])


def git_toplevel(*, repo_root: Path) -> str:
    """返回当前 worktree 的根目录。"""
    return _run_git(repo_root, ["rev-parse", "--show-toplevel"])


def git_changes(*, repo_root: Path) -> list[GitChange]:
    """返回相对 HEAD 的结构化逐文件变更，不使用易漂移的文本路径投影。"""
    status_output = _run_git(
        repo_root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        max_output_chars=None,
    )
    if status_output.startswith("Error:"):
        if not (repo_root / ".git").exists():
            return []
        raise RuntimeError(status_output)
    head = _run_git(repo_root, ["rev-parse", "--verify", "HEAD"], max_output_chars=None)
    if head.startswith("Error:"):
        raise RuntimeError(head)

    records = [] if status_output == "(no output)" else status_output.split("\0")
    entries: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        path = normalize_repo_relative_path(record[3:], field_name="git change path")
        old_path = None
        if "R" in status or "C" in status:
            if index >= len(records) or not records[index]:
                raise RuntimeError("git status returned an incomplete rename/copy record")
            old_path = normalize_repo_relative_path(records[index], field_name="git old path")
            index += 1
        entries.append((status, path, old_path))

    tracked_stats = (
        _tracked_numstat(repo_root)
        if any(status != "??" for status, _, _ in entries)
        else {}
    )
    changes: list[GitChange] = []
    for status, path, old_path in entries:
        kind = _change_kind(status)
        if status == "??":
            added, deleted, binary = _untracked_stats(repo_root, path)
        else:
            stat_paths = [path, old_path] if old_path else [path]
            added = sum(tracked_stats.get(item, (0, 0, False))[0] for item in stat_paths if item)
            deleted = sum(tracked_stats.get(item, (0, 0, False))[1] for item in stat_paths if item)
            binary = any(tracked_stats.get(item, (0, 0, False))[2] for item in stat_paths if item)
        changes.append(
            GitChange(
                kind=kind,
                path=path,
                old_path=old_path,
                added_lines=added,
                deleted_lines=deleted,
                binary=binary,
            )
        )
    return sorted(changes, key=lambda item: (item.path, item.old_path or "", item.kind.value))


def _tracked_numstat(repo_root: Path) -> dict[str, tuple[int, int, bool]]:
    output = _run_git(
        repo_root,
        [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--numstat",
            "-z",
            "HEAD",
            "--",
        ],
        max_output_chars=None,
    )
    if output.startswith("Error:"):
        raise RuntimeError(output)
    stats: dict[str, tuple[int, int, bool]] = {}
    if output == "(no output)":
        return stats
    for record in output.split("\0"):
        if not record:
            continue
        parts = record.split("\t", 2)
        if len(parts) != 3:
            raise RuntimeError("git numstat returned an invalid record")
        added_text, deleted_text, raw_path = parts
        path = normalize_repo_relative_path(raw_path, field_name="git numstat path")
        binary = added_text == "-" or deleted_text == "-"
        stats[path] = (
            int(added_text) if added_text.isdigit() else 0,
            int(deleted_text) if deleted_text.isdigit() else 0,
            binary,
        )
    return stats


def _change_kind(status: str) -> GitChangeKind:
    if status in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"} or "U" in status:
        return GitChangeKind.UNMERGED
    if status == "??" or "A" in status:
        return GitChangeKind.ADDED
    if "R" in status:
        return GitChangeKind.RENAMED
    if "C" in status:
        return GitChangeKind.COPIED
    if "D" in status:
        return GitChangeKind.DELETED
    return GitChangeKind.MODIFIED


def _untracked_stats(repo_root: Path, path: str) -> tuple[int, int, bool]:
    target = safe_repo_path(repo_root, path)
    if target.is_symlink() or not target.is_file():
        return 0, 0, True
    try:
        content = target.read_bytes()
    except OSError:
        # A live lock or another unreadable untracked path is conservatively binary.
        # Scope validation can still reject it, while runtime artifacts can be filtered.
        return 0, 0, True
    if b"\0" in content:
        return 0, 0, True
    return len(content.decode("utf-8", errors="replace").splitlines()), 0, False
