"""
系统启动 / agent调用 memory模块
    ↓
初始化 memory 目录
    ↓
扫描已有 memory 文件
    ↓
构建 / 更新 MEMORY.md index
    ↓
根据 query 进行 memory 检索
    ↓
筛选 top-k memory entries
    ↓
拼接 index + relevant memories
    ↓
返回 prompt 增强文本
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

MEMORY_TYPES = {"user", "feedback", "project", "reference"}
MEMORY_INDEX = "MEMORY.md"
DEFAULT_MEMORY_LIMIT_CHARS = 4_000
SENSITIVE_PATTERNS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "bearer ",
    "C:\\Users\\",
    "/home/",
)


@dataclass(frozen=True)
class MemoryEntry:
    name: str
    description: str
    type: str
    filename: str
    body: str


def memory_dir(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "memory"


def memory_index_path(repo_root: Path) -> Path:
    return memory_dir(repo_root) / MEMORY_INDEX


def ensure_memory_store(repo_root: Path) -> Path:
    directory = memory_dir(repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    index = memory_index_path(repo_root)
    if not index.exists():
        index.write_text("# Memory Index\n\n(no memories yet)\n", encoding="utf-8")
    return directory


def write_memory_file(
    repo_root: Path,
    *,
    name: str,
    mem_type: str,
    description: str,
    body: str,
) -> Path:
    """写入单条长期记忆；保存前过滤 secrets、tokens 和用户私有绝对路径。"""
    if mem_type not in MEMORY_TYPES:
        raise ValueError(f"memory type must be one of: {', '.join(sorted(MEMORY_TYPES))}")
    if _has_sensitive_text(name, description, body):
        raise ValueError("memory contains sensitive text")

    ensure_memory_store(repo_root)
    slug = _slugify(name)
    path = memory_dir(repo_root) / f"{slug}.md"
    path.write_text(
        "---\n"
        f"name: {name.strip()}\n"
        f"description: {description.strip()}\n"
        f"type: {mem_type}\n"
        "---\n\n"
        f"{body.strip()}\n",
        encoding="utf-8",
    )
    rebuild_memory_index(repo_root)
    return path


def rebuild_memory_index(repo_root: Path) -> Path:
    ensure_memory_store(repo_root)
    entries = list_memory_files(repo_root)
    lines = ["# Memory Index", ""]
    if not entries:
        lines.append("(no memories yet)")
    else:
        for entry in entries:
            lines.append(f"- [{entry.name}]({entry.filename}) - {entry.description} ({entry.type})")
    index = memory_index_path(repo_root)
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def list_memory_files(repo_root: Path) -> list[MemoryEntry]:
    directory = memory_dir(repo_root)
    if not directory.exists():
        return []

    entries: list[MemoryEntry] = []
    for path in sorted(directory.glob("*.md")):
        if path.name == MEMORY_INDEX:
            continue
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", path.stem)
        description = meta.get("description", "")
        mem_type = meta.get("type", "reference")
        entries.append(
            MemoryEntry(
                name=name,
                description=description,
                type=mem_type,
                filename=path.name,
                body=body.strip(),
            )
        )
    return entries


def load_memory_index(repo_root: Path, *, limit_chars: int = DEFAULT_MEMORY_LIMIT_CHARS) -> str:
    ensure_memory_store(repo_root)
    text = memory_index_path(repo_root).read_text(encoding="utf-8")
    return text[:limit_chars]


def load_relevant_memories(
    repo_root: Path,
    query: str,
    *,
    max_items: int = 5,
    limit_chars: int = DEFAULT_MEMORY_LIMIT_CHARS,
) -> str:
    entries = select_relevant_memories(repo_root, query, max_items=max_items)
    if not entries:
        return "(no relevant memories)"

    chunks: list[str] = []
    total = 0
    for entry in entries:
        chunk = f"## {entry.name}\n{entry.body}\n"
        if total + len(chunk) > limit_chars:
            remaining = max(0, limit_chars - total)
            if remaining:
                chunks.append(chunk[:remaining])
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n".join(chunks).strip()


def select_relevant_memories(repo_root: Path, query: str, *, max_items: int = 5) -> list[MemoryEntry]:
    words = _keywords(query)
    scored: list[tuple[int, MemoryEntry]] = []
    for entry in list_memory_files(repo_root):
        haystack = f"{entry.name} {entry.description} {entry.body}".lower()
        score = sum(1 for word in words if word in haystack)
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [entry for _score, entry in scored[:max_items]]


def extract_repo_memories(repo_root: Path) -> list[Path]:
    """从仓库标记提取稳定贡献知识；只保存相对路径和通用命令。"""
    ensure_memory_store(repo_root)
    written: list[Path] = []

    if (repo_root / "pyproject.toml").exists():
        written.append(
            _write_if_missing(
                repo_root,
                name="python-test-command",
                mem_type="project",
                description="Use pytest for this Python project when tests are needed.",
                body="Suggested test command: `python -m pytest tests`.",
            )
        )
    if (repo_root / "package.json").exists():
        written.append(
            _write_if_missing(
                repo_root,
                name="javascript-test-command",
                mem_type="project",
                description="Inspect package scripts before running JavaScript tests.",
                body="Suggested workflow: read `package.json`, then run the relevant npm script.",
            )
        )
    if any(repo_root.glob("CONTRIBUTING*")):
        written.append(
            _write_if_missing(
                repo_root,
                name="contribution-guide",
                mem_type="reference",
                description="Read the contribution guide before editing.",
                body="Contribution guidance exists in `CONTRIBUTING*`; read it before making changes.",
            )
        )
    if any((repo_root / ".github").glob("pull_request_template*")):
        written.append(
            _write_if_missing(
                repo_root,
                name="pull-request-template",
                mem_type="reference",
                description="This project has a pull request template.",
                body="Check `.github/pull_request_template*` before drafting the PR body.",
            )
        )

    return [path for path in written if path.exists()]


def memory_prompt(repo_root: Path, *, query: str = "", limit_chars: int = DEFAULT_MEMORY_LIMIT_CHARS) -> str:
    extract_repo_memories(repo_root)
    index = load_memory_index(repo_root, limit_chars=limit_chars)
    relevant = load_relevant_memories(repo_root, query, limit_chars=limit_chars)
    text = (
        "Memory index:\n"
        f"{index}\n"
        "Relevant memory details:\n"
        f"{relevant}"
    )
    return text[:limit_chars]


def _write_if_missing(
    repo_root: Path,
    *,
    name: str,
    mem_type: str,
    description: str,
    body: str,
) -> Path:
    path = memory_dir(repo_root) / f"{_slugify(name)}.md"
    if path.exists():
        return path
    return write_memory_file(repo_root, name=name, mem_type=mem_type, description=description, body=body)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw
    meta: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip("\"'")
    return meta, raw[end + len("\n---") :].lstrip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "memory"


def _keywords(query: str) -> set[str]:
    return {word for word in re.findall(r"[a-zA-Z0-9_-]{3,}", query.lower())}


def _has_sensitive_text(*values: str) -> bool:
    lowered = "\n".join(values).lower()
    return any(pattern.lower() in lowered for pattern in SENSITIVE_PATTERNS)
