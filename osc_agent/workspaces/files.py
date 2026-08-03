"""提供仓库边界内的原子文件写入、编辑和路径匹配。"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
import threading

from osc_agent.workspaces.path_policy import safe_repo_path


_FILE_WRITE_LOCK = threading.RLock()


def write_file(*, repo_root: Path, path: str, content: str) -> int:
    """原子写入 UTF-8 文件并返回写入字符数。"""

    with _FILE_WRITE_LOCK:
        target = safe_repo_path(repo_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, content)
    return len(content)


def edit_file(
    *,
    repo_root: Path,
    path: str,
    old_text: str,
    new_text: str,
) -> int:
    """只替换第一次精确匹配并返回替换次数。"""

    with _FILE_WRITE_LOCK:
        target = safe_repo_path(repo_root, path)
        text = target.read_text(encoding="utf-8")
        if old_text not in text:
            raise ValueError(f"old_text not found in {path}")
        _atomic_write_text(target, text.replace(old_text, new_text, 1))
    return 1


def glob_files(*, repo_root: Path, pattern: str) -> list[str]:
    """返回仓库边界内匹配文件的有序相对路径。"""

    root = repo_root.resolve()
    matches: list[str] = []
    for path in root.glob(pattern):
        resolved = path.resolve()
        if resolved.is_file() and (resolved == root or root in resolved.parents):
            matches.append(resolved.relative_to(root).as_posix())
    return sorted(matches)


def _atomic_write_text(target: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            temporary.chmod(stat.S_IMODE(target.stat().st_mode))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
