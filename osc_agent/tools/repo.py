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
import os
import re
from dataclasses import dataclass
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

DEFAULT_INDEX_MAX_FILES = 20_000
DEFAULT_INDEX_MAX_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class RepositoryIndex:
    root: Path
    paths: tuple[Path, ...]
    total_bytes: int
    truncated: bool

    @classmethod
    def build(
        cls,
        repo_root: Path,
        *,
        max_files: int = DEFAULT_INDEX_MAX_FILES,
        max_bytes: int = DEFAULT_INDEX_MAX_BYTES,
    ) -> "RepositoryIndex":
        root = repo_root.resolve()
        paths: list[Path] = []
        total_bytes = 0
        truncated = False
        for current, dirs, names in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in _SKIP_DIRS)
            for name in sorted(names):
                path = Path(current) / name
                if not _is_safe_repo_file(root, path):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if len(paths) >= max_files or total_bytes + size > max_bytes:
                    truncated = True
                    return cls(root, tuple(paths), total_bytes, truncated)
                paths.append(path)
                total_bytes += size
        return cls(root, tuple(paths), total_bytes, truncated)

    def files(self, *suffixes: str) -> tuple[Path, ...]:
        if not suffixes:
            return self.paths
        return tuple(path for path in self.paths if path.suffix in suffixes)


def inspect_repo(*, repo_root: Path) -> str:
    """扫描贡献任务最常用的入口文件，形成轻量项目地图。"""
    root = repo_root.resolve()
    lines = [f"Repository: {root}"]

    key_patterns = ["README*", "CONTRIBUTING*", "pyproject.toml", "package.json"]
    key_files: list[str] = []
    for pattern in key_patterns:
        for path in root.glob(pattern):
            if _is_safe_repo_file(root, path):
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


def repo_tree(*, repo_root: Path, depth: int = 3, index: RepositoryIndex | None = None) -> str:
    """生成固定深度目录树；跳过缓存和 VCS 目录，避免分析输出被噪声淹没。"""
    root = repo_root.resolve()
    max_depth = max(1, int(depth))
    lines = [root.name + "/"]
    active_index = index or RepositoryIndex.build(root)
    entries: dict[tuple[str, ...], bool] = {}
    for path in active_index.paths:
        parts = path.relative_to(root).parts
        for level in range(1, min(len(parts), max_depth + 1)):
            entries[parts[:level]] = True
        if len(parts) <= max_depth:
            entries[parts] = False
    for parts, is_dir in sorted(entries.items()):
        indent = "  " * (len(parts) - 1)
        lines.append(f"{indent}- {parts[-1]}{'/' if is_dir else ''}")
    if active_index.truncated:
        lines.append("- (repository index truncated by analysis budget)")
    return "\n".join(lines)


def detect_entrypoints(*, repo_root: Path, index: RepositoryIndex | None = None) -> list[str]:
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
    for path in (index or RepositoryIndex.build(root)).paths:
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if _is_safe_repo_file(root, path) and path.name in names:
            matches.append(path.relative_to(root).as_posix())
    return matches


def find_functions(
    *,
    repo_root: Path,
    query: str | None = None,
    index: RepositoryIndex | None = None,
) -> list[dict[str, str]]:
    """轻量提取 Python/TS/JS 函数和类名；用于把分析结论定位到具体代码符号。"""
    root = repo_root.resolve()
    results: list[dict[str, str]] = []
    needle = (query or "").lower()
    for path in (index or RepositoryIndex.build(root)).paths:
        if not _is_safe_repo_file(root, path) or any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix == ".py":
            results.extend(_python_symbols(root, path, needle))
        elif path.suffix in {".ts", ".tsx", ".js", ".jsx"}:
            results.extend(_text_symbols(root, path, needle))
    return results


def analyze_architecture_dimensions(
    *,
    repo_root: Path,
    index: RepositoryIndex | None = None,
) -> list[dict[str, str]]:
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
            symbols.extend(find_functions(repo_root=repo_root, query=keyword, index=index))
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
    index = RepositoryIndex.build(repo_root)
    return {
        "overview": inspect_repo(repo_root=repo_root),
        "tree": repo_tree(repo_root=repo_root, depth=3, index=index),
        "entrypoints": detect_entrypoints(repo_root=repo_root, index=index),
        "repository_profile": detect_repository_profile(repo_root=repo_root, index=index),
        "python_analysis": analyze_python_repository(repo_root=repo_root, index=index),
        "architecture_dimensions": analyze_architecture_dimensions(repo_root=repo_root, index=index),
        "index_metadata": {
            "file_count": len(index.paths),
            "total_bytes": index.total_bytes,
            "truncated": index.truncated,
        },
        "symbols": {
            "planning": find_functions(repo_root=repo_root, query="plan", index=index)[:10],
            "task": find_functions(repo_root=repo_root, query="task", index=index)[:10],
            "tool": find_functions(repo_root=repo_root, query="tool", index=index)[:10],
            "context": find_functions(repo_root=repo_root, query="context", index=index)[:10],
            "trace": find_functions(repo_root=repo_root, query="trace", index=index)[:10],
        },
    }


def detect_repository_profile(*, repo_root: Path, index: RepositoryIndex | None = None) -> dict[str, Any]:
    root = repo_root.resolve()
    python_files = [
        path
        for path in (index or RepositoryIndex.build(root)).files(".py")
        if _is_safe_repo_file(root, path) and not any(part in _SKIP_DIRS for part in path.relative_to(root).parts)
    ]
    marker_text = ""
    for name in ("pyproject.toml", "requirements.txt", "README.md"):
        path = root / name
        if _is_safe_repo_file(root, path):
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


def analyze_python_repository(
    *,
    repo_root: Path,
    max_call_depth: int = 2,
    index: RepositoryIndex | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    modules: dict[str, str] = {}
    imports: dict[str, list[str]] = {}
    definitions: list[dict[str, Any]] = []
    references: dict[str, list[dict[str, Any]]] = {}
    calls: dict[str, set[str]] = {}

    syntax_errors: list[dict[str, Any]] = []
    for path in (index or RepositoryIndex.build(root)).files(".py"):
        relative = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in relative.parts) or not _is_safe_repo_file(root, path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            syntax_errors.append({"file": relative.as_posix(), "line": exc.lineno, "message": exc.msg})
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
        "syntax_errors": syntax_errors,
    }


def analyze_issue_code_candidates(
    *,
    repo_root: Path,
    issue: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """把 Issue 中的路径、标识符和错误词映射到静态代码证据，不执行目标项目。"""
    root = repo_root.resolve()
    issue_text = f"{issue.get('title', '')}\n{issue.get('body', '')}"
    lowered = issue_text.casefold()
    terms = _issue_search_terms(issue_text)
    explicit_paths = {
        value.replace("\\", "/").lstrip("./")
        for value in re.findall(r"[A-Za-z0-9_./\\-]+\.py", issue_text)
    }
    index = RepositoryIndex.build(root)
    analysis = analyze_python_repository(repo_root=root, index=index)
    candidates: dict[str, dict[str, Any]] = {}

    for definition in analysis.get("definitions") or []:
        relative = str(definition.get("file") or "")
        name = str(definition.get("name") or "")
        score = 0
        reasons: list[str] = []
        if relative in explicit_paths:
            score += 100
            reasons.append("Issue references this file")
        if name and name.casefold() in lowered:
            score += 60
            reasons.append(f"Issue references symbol {name}")
        matched_terms = sorted(term for term in terms if term in name.casefold())
        if matched_terms:
            score += 15 * len(matched_terms)
            reasons.append(f"Symbol matches Issue terms: {', '.join(matched_terms[:5])}")
        if score:
            entry = candidates.setdefault(relative, {"file": relative, "score": 0, "reasons": [], "symbols": []})
            entry["score"] += score
            entry["reasons"].extend(reasons)
            entry["symbols"].append(
                {
                    "name": name,
                    "kind": definition.get("kind"),
                    "line_range": definition.get("line_range"),
                    "content_hash": definition.get("content_hash"),
                }
            )

    for relative in explicit_paths:
        path = root / relative
        if _is_safe_repo_file(root, path):
            entry = candidates.setdefault(relative, {"file": relative, "score": 0, "reasons": [], "symbols": []})
            entry["score"] += 100
            entry["reasons"].append("Issue references this file")

    for path in index.files(".py"):
        if not _is_safe_repo_file(root, path):
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in _SKIP_DIRS for part in Path(relative).parts):
            continue
        content = path.read_text(encoding="utf-8", errors="replace").casefold()
        matched_terms = sorted(term for term in terms if term in content)
        if not matched_terms:
            continue
        entry = candidates.setdefault(relative, {"file": relative, "score": 0, "reasons": [], "symbols": []})
        entry["score"] += min(40, 5 * len(matched_terms))
        entry["reasons"].append(f"File content matches Issue terms: {', '.join(matched_terms[:5])}")

    for relative, entry in candidates.items():
        path = root / relative
        if not _is_safe_repo_file(root, path):
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        matches = []
        for number, line in enumerate(lines, start=1):
            matched = [term for term in terms if term in line.casefold()]
            if matched:
                matches.append({"line": number, "terms": sorted(set(matched))[:5]})
            if len(matches) >= 5:
                break
        entry["matched_lines"] = matches
        entry["reasons"] = sorted(set(entry["reasons"]))

    return sorted(candidates.values(), key=lambda item: (-int(item["score"]), str(item["file"])))[: max(1, int(limit))]


def _issue_search_terms(text: str) -> set[str]:
    stop_words = {
        "about", "actual", "after", "agent", "before", "behavior", "error", "expected",
        "feature", "issue", "python", "should", "steps", "test", "tests", "this", "when", "with",
    }
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text)
        if token.casefold() not in stop_words
    }
    return set(sorted(terms, key=lambda item: (-len(item), item))[:30])


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


def _is_safe_repo_file(root: Path, path: Path) -> bool:
    """只允许读取解析后仍位于仓库内的普通文件。"""
    try:
        resolved = path.resolve()
        return path.is_file() and (resolved == root or root in resolved.parents)
    except (OSError, RuntimeError):
        return False


def _python_symbols(root: Path, path: Path, needle: str) -> list[dict[str, str]]:
    if not _is_safe_repo_file(root, path):
        return []
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
    if not _is_safe_repo_file(root, path):
        return []
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
