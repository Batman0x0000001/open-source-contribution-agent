"""
模型调用 inspect_repo
        ↓
扫描仓库根目录
        ↓
寻找关键配置文件
        ↓
寻找测试目录
        ↓
生成仓库概览（项目地图）
"""

from __future__ import annotations

from pathlib import Path

REPO_TOOLS = [
    {
        "name": "inspect_repo",
        "description": "Build a small map of key repository files and test directories.",
        "input_schema": {"type": "object", "properties": {}},
    }
]


def inspect_repo(*, repo_root: Path) -> str:
    """扫描贡献任务最常用的入口文件，形成轻量项目地图。"""
    root = repo_root.resolve()
    lines = [f"Repository: {root}"]

    key_patterns = ["README*", "CONTRIBUTING*", "pyproject.toml", "package.json"]
    key_files: list[str] = []
    for pattern in key_patterns:
        for path in root.glob(pattern):
            if path.is_file():
                key_files.append(path.relative_to(root).as_posix())

    test_entries = []
    for name in ("tests", "test"):
        path = root / name
        if path.exists():
            test_entries.append(path.relative_to(root).as_posix())

    lines.append("Key files:")
    lines.extend(f"- {item}" for item in sorted(set(key_files)) or ["(none)"])
    lines.append("Test entries:")
    lines.extend(f"- {item}" for item in sorted(set(test_entries)) or ["(none)"])
    return "\n".join(lines)
