"""
Agent 执行事件
      ↓
append_trace()
      ↓
生成一条 JSON
      ↓
追加到 session.jsonl
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import threading
import hashlib
from itertools import islice
import re
from typing import Any

TRACE_WRITE_MODE = "a"
_trace_lock = threading.Lock()


def trace_path(repo_root: Path) -> Path:
    """统一计算 trace 文件位置，便于后续阶段恢复和审计。"""
    return repo_root / ".osc_agent" / "traces" / "session.jsonl"


def append_trace(repo_root: Path, event: str, payload: dict[str, Any]) -> None:
    """把单条审计事件追加到 JSONL；每行都是独立 JSON，方便流式读取。"""
    path = trace_path(repo_root)
    record = {
        "ts": datetime.now(ZoneInfo("Asia/ShangHai")).isoformat(),
        "event": event,
        **payload,
    }
    with _trace_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(TRACE_WRITE_MODE, encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def preview(value: Any, limit: int = 200) -> str:
    """生成 trace 预览，避免大输出直接塞进审计日志。"""
    return str(value)[:limit]


_SENSITIVE_KEY = re.compile(r"(?:token|secret|password|api[_ -]?key|authorization)", re.I)
_INLINE_SECRET = re.compile(
    r"(?i)\b(token|secret|password|api[_ -]?key|authorization)\b"
    r"[\"']?\s*[=:]\s*[\"']?(?:bearer\s+)?[^\s,\"';}]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_KNOWN_API_TOKEN = re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{8,}\b", re.I)
MAX_ARGUMENT_STRING_CHARS = 2_000
MAX_ARGUMENT_COLLECTION_ITEMS = 20
MAX_ARGUMENT_DEPTH = 4
MAX_SANITIZED_ARGUMENT_CHARS = 8_000


def sanitize_trace_text(value: Any, *, limit: int = 1_000) -> str:
    """对将进入 Trace 的自由文本统一脱敏并限制长度。"""
    text = _INLINE_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", str(value))
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    text = _KNOWN_API_TOKEN.sub("[REDACTED]", text)
    return text[: max(0, limit)]


def sanitize_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """保留可追溯信息，同时递归移除凭据、源码正文和超大参数。"""
    sanitized = _sanitize_mapping(tool_name, arguments, depth=0)
    encoded = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) <= MAX_SANITIZED_ARGUMENT_CHARS:
        return sanitized
    return {
        "preview": encoded[:MAX_SANITIZED_ARGUMENT_CHARS],
        "chars": len(encoded),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "truncated": True,
    }


def _sanitize_mapping(tool_name: str, values: dict[Any, Any], *, depth: int) -> dict[str, Any]:
    if depth >= MAX_ARGUMENT_DEPTH:
        return {"truncated": True, "reason": "maximum argument depth reached"}
    sanitized: dict[str, Any] = {}
    for raw_key, value in islice(values.items(), MAX_ARGUMENT_COLLECTION_ITEMS):
        key = str(raw_key)
        if _SENSITIVE_KEY.search(key):
            sanitized[key] = "[REDACTED]"
        elif tool_name in {"write_file", "edit_file"} and key in {"content", "old_text", "new_text"}:
            text = str(value)
            sanitized[key] = {
                "chars": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        else:
            sanitized[key] = _sanitize_value(tool_name, value, depth=depth + 1)
    if len(values) > MAX_ARGUMENT_COLLECTION_ITEMS:
        sanitized["__truncated_items__"] = len(values) - MAX_ARGUMENT_COLLECTION_ITEMS
    return sanitized


def _sanitize_value(tool_name: str, value: Any, *, depth: int) -> Any:
    if isinstance(value, dict):
        return _sanitize_mapping(tool_name, value, depth=depth)
    if isinstance(value, (list, tuple)):
        if depth >= MAX_ARGUMENT_DEPTH:
            return {"truncated": True, "reason": "maximum argument depth reached"}
        items = [
            _sanitize_value(tool_name, item, depth=depth + 1)
            for item in value[:MAX_ARGUMENT_COLLECTION_ITEMS]
        ]
        if len(value) > MAX_ARGUMENT_COLLECTION_ITEMS:
            items.append({"truncated_items": len(value) - MAX_ARGUMENT_COLLECTION_ITEMS})
        return items
    if isinstance(value, str):
        return sanitize_trace_text(value, limit=MAX_ARGUMENT_STRING_CHARS)
    return value
