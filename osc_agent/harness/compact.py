"""
messages 变长
    ↓
apply_compaction()
    ↓
压缩超大 tool_result
    ↓
裁剪中间旧消息
    ↓
压缩较早工具输出
    ↓
如果仍然太大，整体 compact 成摘要
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from osc_agent.harness.trace import append_trace, preview

MAX_MESSAGES = 50
KEEP_HEAD_MESSAGES = 3
KEEP_RECENT_TOOL_RESULTS = 3
MICRO_COMPACT_MIN_CHARS = 200
TOOL_RESULT_BUDGET_CHARS = 200_000
PERSIST_PREVIEW_CHARS = 2_000
AUTO_COMPACT_THRESHOLD_CHARS = 120_000
REACTIVE_TAIL_MESSAGES = 5

COMPACT_TOOL = {
    "name": "compact",
    "description": "Summarize and compact the active conversation history.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "additionalProperties": False,
    },
}


def estimate_size(messages: list[dict[str, Any]]) -> int:
    """用 JSON 字符长度近似上下文大小，避免引入 tokenizer 依赖。"""
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def tool_result_budget(
    messages: list[dict[str, Any]],
    *,
    repo_root: Path,
    max_chars: int = TOOL_RESULT_BUDGET_CHARS,
) -> list[dict[str, Any]]:
    if not messages:
        return messages

    last_message = messages[-1]
    blocks = _tool_result_blocks(last_message)
    total = sum(len(str(block.get("content", ""))) for block in blocks)
    if total <= max_chars:
        return messages

    ranked = sorted(blocks, key=lambda block: len(str(block.get("content", ""))), reverse=True)
    for block in ranked:
        #循环里每压缩一个 block，total 都会重新计算
        if total <= max_chars:
            break
        original = str(block.get("content", ""))
        persisted = persist_tool_result(
            repo_root,
            str(block.get("tool_use_id", "unknown")),
            original,
        )
        block["content"] = (
            "<persisted-output>\n"
            f"path: {persisted.relative_to(repo_root).as_posix()}\n"
            f"preview:\n{preview(original, PERSIST_PREVIEW_CHARS)}"
        )
        total = sum(len(str(item.get("content", ""))) for item in blocks)

    append_trace(repo_root, "compact_tool_result_budget", {"remaining_chars": total})
    return messages


def snip_compact(messages: list[dict[str, Any]], max_messages: int = MAX_MESSAGES) -> list[dict[str, Any]]:
    if len(messages) <= max_messages:
        return messages

    head_end = min(KEEP_HEAD_MESSAGES, len(messages))
    tail_start = len(messages) - (max_messages - head_end)

    # 切口不能把 assistant tool_use 和紧随其后的 user tool_result 拆开。
    while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
        head_end += 1
    if 0 < tail_start < len(messages) and _is_tool_result_message(messages[tail_start]):
        tail_start -= 1

    snipped = max(0, tail_start - head_end)
    placeholder = {
        "role": "user",
        "content": f"[snipped {snipped} messages from conversation middle]",
    }
    return messages[:head_end] + [placeholder] + messages[tail_start:]


def micro_compact(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
    min_chars: int = MICRO_COMPACT_MIN_CHARS,
) -> list[dict[str, Any]]:
    tool_results = _collect_tool_results(messages)
    if len(tool_results) <= keep_recent:
        return messages

    for block in tool_results[:-keep_recent]:
        content = str(block.get("content", ""))
        if len(content) > min_chars and not content.startswith("<persisted-output>"):
            block["content"] = f"[Earlier tool result compacted. Preview: {preview(content)}]"
    return messages


def compact_history(
    messages: list[dict[str, Any]],
    *,
    repo_root: Path,
    reason: str = "auto",
) -> list[dict[str, Any]]:
    transcript = write_transcript(repo_root, messages, prefix=reason)
    summary = summarize_history(messages)
    append_trace(
        repo_root,
        "compact_history",
        {"reason": reason, "transcript": transcript.relative_to(repo_root).as_posix()},
    )
    return [
        {
            "role": "user",
            "content": (
                f"[Compacted: {reason}]\n"
                f"Transcript: {transcript.relative_to(repo_root).as_posix()}\n\n"
                f"{summary}"
            ),
        }
    ]


def reactive_compact(messages: list[dict[str, Any]], *, repo_root: Path) -> list[dict[str, Any]]:
    transcript = write_transcript(repo_root, messages, prefix="reactive")
    tail_start = max(0, len(messages) - REACTIVE_TAIL_MESSAGES)
    if 0 < tail_start < len(messages) and _is_tool_result_message(messages[tail_start]):
        tail_start -= 1

    summary = summarize_history(messages[:tail_start])
    append_trace(
        repo_root,
        "reactive_compact",
        {"transcript": transcript.relative_to(repo_root).as_posix(), "tail_messages": len(messages) - tail_start},
    )
    return [
        {
            "role": "user",
            "content": (
                "[Reactive compact]\n"
                f"Transcript: {transcript.relative_to(repo_root).as_posix()}\n\n"
                f"{summary}"
            ),
        },
        *messages[tail_start:],
    ]


def apply_compaction(messages: list[dict[str, Any]], *, repo_root: Path) -> list[dict[str, Any]]:
    compacted = tool_result_budget(messages, repo_root=repo_root)
    compacted = snip_compact(compacted)
    compacted = micro_compact(compacted)
    if estimate_size(compacted) > AUTO_COMPACT_THRESHOLD_CHARS:
        compacted = compact_history(compacted, repo_root=repo_root, reason="auto")
    return compacted


def persist_tool_result(repo_root: Path, tool_use_id: str, content: str) -> Path:
    directory = repo_root / ".osc_agent" / "tool-results"
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tool_use_id) or "unknown"
    path = directory / f"{safe_id}.txt"
    counter = 1
    while path.exists():
        path = directory / f"{safe_id}-{counter}.txt"
        counter += 1
    path.write_text(content, encoding="utf-8")
    return path


def write_transcript(repo_root: Path, messages: list[dict[str, Any]], *, prefix: str) -> Path:
    directory = repo_root / ".osc_agent" / "transcripts"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = directory / f"{prefix}-{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    return path


def summarize_history(messages: list[dict[str, Any]]) -> str:
    """生成确定性的轻量摘要；生产级 LLM 摘要留给后续更完整的模型封装。"""
    user_messages = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    tool_results = _collect_tool_results(messages)
    return "\n".join(
        [
            "Summary:",
            f"- Messages before compact: {len(messages)}",
            f"- User messages: {len(user_messages)}",
            f"- Assistant messages: {len(assistant_messages)}",
            f"- Tool results: {len(tool_results)}",
            f"- Recent user context: {preview(user_messages[-1] if user_messages else '(none)', 500)}",
        ]
    )


def _collect_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        results.extend(_tool_result_blocks(message))
    return results


def _tool_result_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    return bool(_tool_result_blocks(message))
