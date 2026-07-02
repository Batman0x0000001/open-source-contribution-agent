"""
模型请求文件工具
      ↓
调用对应函数
      ↓
检查路径是否在仓库内
      ↓
执行读 / 写 / 编辑 / 搜索
      ↓
返回结果给 Agent
"""

from __future__ import annotations

from pathlib import Path

from osc_agent.harness.permissions import check_file_write, format_blocked, safe_repo_path

FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read text from a file inside the target repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "default": 20_000},
                "offset": {"type": "integer", "default": 0},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a file inside the target repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace one occurrence of old_text in a file inside the target repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "List repository files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]


def read_file(*, repo_root: Path, path: str, limit: int = 20_000, offset: int = 0) -> str:
    """读取文件片段，offset/limit 用来控制大文件进入上下文的大小。"""
    try:
        target = safe_repo_path(repo_root, path)
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return f"Error: {exc}"

    start = max(0, int(offset))
    end = start + max(0, int(limit))
    return text[start:end]


def write_file(*, repo_root: Path, path: str, content: str, enforce_permissions: bool = True) -> str:
    """写入 repo 内文件；父目录不存在时按常见编辑工具行为创建。"""
    if enforce_permissions:
        decision = check_file_write(path, content)
        if not decision.allowed:
            return format_blocked(decision)

    try:
        target = safe_repo_path(repo_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    return f"Wrote {path}"


def edit_file(
    *,
    repo_root: Path,
    path: str,
    old_text: str,
    new_text: str,
    enforce_permissions: bool = True,
) -> str:
    """只替换第一次匹配，防止模型一次调用意外改动多个位置。"""
    if enforce_permissions:
        decision = check_file_write(path, new_text)
        if not decision.allowed:
            return format_blocked(decision)

    try:
        target = safe_repo_path(repo_root, path)
        text = target.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: old_text not found in {path}"
        target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return f"Error: {exc}"
    return f"Edited {path}"


def glob_files(*, repo_root: Path, pattern: str) -> str:
    """返回匹配文件的相对路径列表，便于模型快速建立项目视图。"""
    try:
        root = repo_root.resolve()
        matches = []
        for path in root.glob(pattern):
            resolved = path.resolve()
            if resolved.is_file() and (resolved == root or root in resolved.parents):
                matches.append(resolved.relative_to(root).as_posix())
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"

    return "\n".join(sorted(matches)) or "(no matches)"
