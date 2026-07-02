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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_WRITE_MODE = "a"


def trace_path(repo_root: Path) -> Path:
    """统一计算 trace 文件位置，便于后续阶段恢复和审计。"""
    return repo_root / ".osc_agent" / "traces" / "session.jsonl"


def append_trace(repo_root: Path, event: str, payload: dict[str, Any]) -> None:
    """把单条审计事件追加到 JSONL；每行都是独立 JSON，方便流式读取。"""
    path = trace_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open(TRACE_WRITE_MODE, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def preview(value: Any, limit: int = 200) -> str:
    """生成 trace 预览，避免大输出直接塞进审计日志。"""
    return str(value)[:limit]
