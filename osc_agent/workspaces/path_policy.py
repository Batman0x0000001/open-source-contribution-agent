"""校验仓库相对路径并阻止越界文件访问。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def safe_repo_path(repo_root: Path, path: str) -> Path:
    root = repo_root.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes repository: {path}")
    return target


def normalize_repo_relative_path(value: Any, *, field_name: str = "path") -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = value.replace("\\", "/")
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field_name} must contain repository-relative paths")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"{field_name} contains a non-canonical repository path: {value}")
    return normalized


def normalize_repo_relative_pattern(value: Any, *, field_name: str = "pattern") -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = value.replace("\\", "/")
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field_name} must contain repository-relative patterns")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"{field_name} contains a non-canonical repository pattern: {value}")
    return normalized


def repo_path_matches(path: str, pattern: str) -> bool:
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
