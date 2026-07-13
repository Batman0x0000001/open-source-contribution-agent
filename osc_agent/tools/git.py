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

import os
import subprocess
from pathlib import Path

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
_AGENT_METADATA_PREFIX = ".osc_agent/"


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


def git_changed_files(*, repo_root: Path) -> list[str]:
    """返回逐文件变更列表，包含未跟踪目录中的每个文件。"""
    output = _run_git(
        repo_root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        max_output_chars=None,
    )
    if output.startswith("Error:"):
        if not (repo_root / ".git").exists():
            return []
        raise RuntimeError(output)
    if output == "(no output)":
        return []
    files: list[str] = []
    records = output.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        files.append(record[3:].replace("\\", "/"))
        if "R" in status or "C" in status:
            index += 1  # -z 格式会在目标路径后附带原路径。
    return files


def git_diff_numstat(*, repo_root: Path) -> tuple[int, int]:
    """统计相对 HEAD 的增删行，并补充未跟踪文本文件的新增行。"""
    output = _run_git(
        repo_root,
        ["diff", "--no-ext-diff", "--no-textconv", "--numstat", "HEAD", "--"],
        max_output_chars=None,
    )
    if output.startswith("Error:"):
        if not (repo_root / ".git").exists():
            return 0, 0
        raise RuntimeError(output)
    if output == "(no output)":
        output = ""
    added = deleted = 0
    for line in output.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        if parts[0].isdigit():
            added += int(parts[0])
        if parts[1].isdigit():
            deleted += int(parts[1])
    status = _run_git(
        repo_root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        max_output_chars=None,
    )
    if status.startswith("Error:"):
        if not (repo_root / ".git").exists():
            return added, deleted
        raise RuntimeError(status)
    if status != "(no output)":
        for record in status.split("\0"):
            if not record.startswith("?? "):
                continue
            relative = record[3:].replace("\\", "/")
            if relative.startswith(_AGENT_METADATA_PREFIX):
                continue
            target = repo_root / relative
            if target.is_symlink() or not target.is_file():
                continue
            try:
                content = target.read_bytes()
                if b"\0" not in content:
                    added += len(content.decode("utf-8", errors="replace").splitlines())
            except OSError:
                continue
    return added, deleted
