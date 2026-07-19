from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def safe_repo_path(repo_root: Path, path: str) -> Path:
    """解析目标路径，并确保最终路径没有逃出仓库根目录。"""
    root = repo_root.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes repository: {path}")
    return target


def normalize_repo_relative_path(value: Any, *, field_name: str = "path") -> str:
    """将具体仓库路径规范为 POSIX 相对路径，并拒绝含糊或逃逸形式。"""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = value.replace("\\", "/")
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field_name} must contain repository-relative paths")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field_name} contains a non-canonical repository path: {value}")
    return "/".join(parts)


def normalize_repo_relative_pattern(value: Any, *, field_name: str = "pattern") -> str:
    """规范仓库相对 Glob Pattern，同时禁止绝对路径和父目录逃逸。"""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = value.replace("\\", "/")
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field_name} must contain repository-relative patterns")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field_name} contains a non-canonical repository pattern: {value}")
    return "/".join(parts)


def repo_path_matches(path: str, pattern: str) -> bool:
    """使用统一的 POSIX 语义匹配仓库路径；directory/** 表示该目录下任意深度。"""
    normalized_path = normalize_repo_relative_path(path)
    normalized_pattern = normalize_repo_relative_pattern(pattern)
    if normalized_pattern.endswith("/**") and not any(
        marker in normalized_pattern[:-3] for marker in ("*", "?", "[")
    ):
        directory = normalized_pattern[:-3].rstrip("/")
        return normalized_path == directory or normalized_path.startswith(f"{directory}/")
    candidate = PurePosixPath(normalized_path)
    return candidate.match(normalized_pattern) or (
        normalized_pattern.startswith("**/") and candidate.match(normalized_pattern[3:])
    )


def repo_path_is_within(path: str, directory: str) -> bool:
    normalized_path = normalize_repo_relative_path(path)
    normalized_directory = normalize_repo_relative_path(directory, field_name="directory")
    return normalized_path.startswith(f"{normalized_directory}/")


def repository_boundary_summary() -> str:
    """返回与 safe_repo_path 一致的仓库路径边界说明。"""
    return "File paths must resolve inside the target repository; repository path escapes are blocked."
