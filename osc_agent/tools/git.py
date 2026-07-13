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


def _run_git(repo_root: Path, args: list[str]) -> str:
    """用参数列表调用 git，避免通过 shell 拼接只读 git 命令。"""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Error: {exc}"

    output = completed.stdout or ""
    if completed.returncode != 0:
        output += completed.stderr or ""
    output = output.strip("\r\n")
    return (output or "(no output)")[:MAX_GIT_OUTPUT_CHARS]


def git_status(*, repo_root: Path) -> str:
    """读取当前仓库状态，不执行任何写操作。"""
    return _run_git(repo_root, ["status", "--short"])


def git_diff(*, repo_root: Path) -> str:
    """读取当前工作区 diff，用于后续贡献摘要。"""
    return _run_git(repo_root, ["diff", "--"])


def git_log(*, repo_root: Path, limit: int = 5) -> str:
    """读取最近提交，limit 做最小夹取避免过长输出。"""
    safe_limit = min(max(int(limit), 1), 50)
    return _run_git(repo_root, ["log", f"-{safe_limit}", "--oneline"])


def git_head(*, repo_root: Path) -> str:
    return _run_git(repo_root, ["rev-parse", "HEAD"])


def git_changed_files(*, repo_root: Path) -> list[str]:
    output = _run_git(repo_root, ["status", "--porcelain"])
    if output.startswith("Error:") or output == "(no output)":
        return []
    files: list[str] = []
    for line in output.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path.strip('"').replace("\\", "/"))
    return files


def git_diff_numstat(*, repo_root: Path) -> tuple[int, int]:
    output = _run_git(repo_root, ["diff", "--numstat", "HEAD", "--"])
    if output.startswith("Error:"):
        return 0, 0
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
    status = _run_git(repo_root, ["status", "--porcelain"])
    if status != "(no output)" and not status.startswith("Error:"):
        for line in status.splitlines():
            if not line.startswith("?? "):
                continue
            relative = line[3:].strip().strip('"')
            target = repo_root / relative
            if not target.is_file():
                continue
            try:
                content = target.read_bytes()
                if b"\0" not in content:
                    added += len(content.decode("utf-8", errors="replace").splitlines())
            except OSError:
                continue
    return added, deleted
