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

    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
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
