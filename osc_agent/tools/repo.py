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

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from osc_agent.skills.registry import suggest_skills_for_repo

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
    lines.append("Suggested skills:")
    lines.extend(f"- {item}" for item in suggest_skills_for_repo(root) or ["(none)"])
    return "\n".join(lines)


def repo_tree(*, repo_root: Path, depth: int = 3) -> str:
    """生成固定深度目录树；跳过缓存和 VCS 目录，避免分析输出被噪声淹没。"""
    root = repo_root.resolve()
    max_depth = max(1, int(depth))
    lines = [root.name + "/"]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) > max_depth or any(part in _SKIP_DIRS for part in parts):
            continue
        indent = "  " * (len(parts) - 1)
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}- {path.name}{suffix}")
    return "\n".join(lines)


def detect_entrypoints(*, repo_root: Path) -> list[str]:
    """按常见命名寻找项目入口文件，供第一步建立项目理解。"""
    root = repo_root.resolve()
    names = {
        "main.py",
        "app.py",
        "cli.py",
        "__main__.py",
        "index.ts",
        "index.js",
        "app.ts",
        "app.js",
    }
    matches: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.name in names:
            matches.append(path.relative_to(root).as_posix())
    return matches


def find_functions(*, repo_root: Path, query: str | None = None) -> list[dict[str, str]]:
    """轻量提取 Python/TS/JS 函数和类名；用于把分析结论定位到具体代码符号。"""
    root = repo_root.resolve()
    results: list[dict[str, str]] = []
    needle = (query or "").lower()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix == ".py":
            results.extend(_python_symbols(root, path, needle))
        elif path.suffix in {".ts", ".tsx", ".js", ".jsx"}:
            results.extend(_text_symbols(root, path, needle))
    return results


def analyze_architecture_dimensions(*, repo_root: Path) -> list[dict[str, str]]:
    """按 7 个 OpenSourcePR 维度给出证据定位；没有证据时显式标记未定位。"""
    dimensions = [
        ("任务规划", ["plan", "task", "todo"]),
        ("多 Agent 协作", ["agent", "team", "worker", "orchestrator"]),
        ("上下文管理策略", ["context", "compact", "memory", "history"]),
        ("Human-in-the-loop 机制", ["permission", "confirm", "approve", "checkpoint", "human"]),
        ("Agent 评估框架", ["eval", "metric", "score", "trajectory"]),
        ("Tool 检索与路由", ["tool", "dispatch", "route", "mcp"]),
        ("Streaming 与中间状态可见性", ["stream", "event", "status", "trace"]),
    ]
    rows: list[dict[str, str]] = []
    for name, keywords in dimensions:
        symbols: list[dict[str, str]] = []
        for keyword in keywords:
            symbols.extend(find_functions(repo_root=repo_root, query=keyword))
            if symbols:
                break
        if symbols:
            first = symbols[0]
            location = f"{first['file']}::{first['name']}"
            current = f"定位到 {first['kind']} {location}"
        else:
            location = "未定位到具体实现"
            current = "未定位到具体实现"
        rows.append(
            {
                "dimension": name,
                "location": location,
                "current": current,
                "gap": "需要人工结合源码确认是否足以支撑贡献方向。",
                "impact": "medium",
                "improvement": "围绕该维度选择最小可审查的改造点，并补充测试或文档证据。",
                "scope": "预计 1-3 个文件，约 50-300 行。",
                "interview_angle": f"体现对{name}的架构分析和渐进式改造能力。",
            }
        )
    return rows


def collect_repo_evidence_pack(*, repo_root: Path) -> dict[str, object]:
    """收集供 LLM 分析消费的结构化仓库证据包。"""
    return {
        "overview": inspect_repo(repo_root=repo_root),
        "tree": repo_tree(repo_root=repo_root, depth=3),
        "entrypoints": detect_entrypoints(repo_root=repo_root),
        "repository_profile": detect_repository_profile(repo_root=repo_root),
        "python_analysis": analyze_python_repository(repo_root=repo_root),
        "architecture_dimensions": analyze_architecture_dimensions(repo_root=repo_root),
        "symbols": {
            "planning": find_functions(repo_root=repo_root, query="plan")[:10],
            "task": find_functions(repo_root=repo_root, query="task")[:10],
            "tool": find_functions(repo_root=repo_root, query="tool")[:10],
            "context": find_functions(repo_root=repo_root, query="context")[:10],
            "trace": find_functions(repo_root=repo_root, query="trace")[:10],
        },
    }


def detect_repository_profile(*, repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    python_files = [path for path in root.rglob("*.py") if not any(part in _SKIP_DIRS for part in path.parts)]
    marker_text = ""
    for name in ("pyproject.toml", "requirements.txt", "README.md"):
        path = root / name
        if path.is_file():
            marker_text += path.read_text(encoding="utf-8", errors="replace")[:20_000].lower()
    agent_markers = sorted(
        marker for marker in ("agent", "llm", "anthropic", "openai", "langchain", "langgraph", "tool use")
        if marker in marker_text or any(marker.replace(" ", "_") in path.name.lower() for path in python_files)
    )
    return {
        "language": "python" if python_files or (root / "pyproject.toml").exists() else "unsupported",
        "agent_llm_markers": agent_markers,
        "supported": bool(python_files and agent_markers),
        "python_file_count": len(python_files),
    }


def analyze_python_repository(*, repo_root: Path, max_call_depth: int = 2) -> dict[str, Any]:
    root = repo_root.resolve()
    modules: dict[str, str] = {}
    imports: dict[str, list[str]] = {}
    definitions: list[dict[str, Any]] = []
    references: dict[str, list[dict[str, Any]]] = {}
    calls: dict[str, set[str]] = {}

    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        rel = relative.as_posix()
        module = ".".join(relative.with_suffix("").parts)
        modules[module] = rel
        imports[rel] = sorted(_python_imports(tree))
        current_function: str | None = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                snippet = "\n".join(text.splitlines()[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)])
                definitions.append(
                    {
                        "file": rel,
                        "name": node.name,
                        "kind": kind,
                        "line_range": [node.lineno, getattr(node, "end_lineno", node.lineno)],
                        "content_hash": hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
                    }
                )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    current_function = f"{rel}::{node.name}"
                    calls[current_function] = {
                        child.func.id
                        for child in ast.walk(node)
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    }
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                references.setdefault(node.id, []).append({"file": rel, "line": node.lineno})

    tests = [path for path in modules.values() if path.startswith("tests/") or Path(path).name.startswith("test_")]
    test_mapping: dict[str, list[str]] = {}
    for definition in definitions:
        if definition["file"] in tests:
            continue
        stem = Path(definition["file"]).stem.lower()
        matches = [test for test in tests if stem in Path(test).stem.lower()]
        if matches:
            test_mapping[definition["file"]] = sorted(set(matches))

    expanded_calls = {
        caller: _expand_calls(targets, calls, max_depth=max(1, int(max_call_depth)))
        for caller, targets in calls.items()
    }
    return {
        "imports": imports,
        "definitions": definitions,
        "references": references,
        "test_mapping": test_mapping,
        "call_expansion": expanded_calls,
    }


def _python_imports(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def _expand_calls(targets: set[str], calls: dict[str, set[str]], *, max_depth: int) -> list[str]:
    known_by_name: dict[str, list[str]] = {}
    for qualified in calls:
        known_by_name.setdefault(qualified.rsplit("::", 1)[-1], []).append(qualified)
    visited: set[str] = set(targets)
    frontier = set(targets)
    for _ in range(max_depth - 1):
        next_frontier: set[str] = set()
        for name in frontier:
            for qualified in known_by_name.get(name, []):
                next_frontier.update(calls.get(qualified, set()))
        next_frontier -= visited
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier
    return sorted(visited)


_SKIP_DIRS = {".git", ".osc_agent", ".pytest_cache", "__pycache__", "node_modules", ".venv", "dist", "build"}


def _python_symbols(root: Path, path: Path, needle: str) -> list[dict[str, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if needle and needle not in node.name.lower() and needle not in path.as_posix().lower():
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            results.append({"file": path.relative_to(root).as_posix(), "name": node.name, "kind": kind})
    return results


def _text_symbols(root: Path, path: Path, needle: str) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    results = []
    pattern = re.compile(r"(?:function\s+|class\s+|const\s+|let\s+|export\s+function\s+)([A-Za-z_$][\w$]*)")
    for match in pattern.finditer(text):
        name = match.group(1)
        if needle and needle not in name.lower() and needle not in path.as_posix().lower():
            continue
        results.append({"file": path.relative_to(root).as_posix(), "name": name, "kind": "symbol"})
    return results
