"""
Lead Agent 调用 spawn_teammate
    ↓
创建 teammate 独立线程
    ↓
teammate 启动自己的 LLM loop
    ↓
每轮先读取 inbox，并路由协议消息
    ↓
LLM 根据 inbox / task / tool_result 决定行动
    ↓
可调用工具：读文件、查仓库、发消息、提交计划审批、认领/完成任务
    ↓
如果无 tool_use 且 autonomous=true，进入 idle_poll
    ↓
idle_poll 周期性检查 inbox 或自动认领可执行 task
    ↓
完成 / 等审批 / 超时 / shutdown 后，把最终结果发回 lead
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from osc_agent.config import Settings
from osc_agent.harness.hooks import HookContext, default_hooks, elapsed_ms
from osc_agent.harness.subagent import READ_ONLY_BASH_PREFIXES
from osc_agent.harness.tasks import (
    CONTRIBUTION_TASK_TOOLS,
    blocking_dependencies,
    claim_task,
    complete_task,
    get_task,
    list_tasks,
    load_all_tasks,
)
from osc_agent.harness.trace import append_trace, preview
from osc_agent.harness.worktree import resolve_task_worktree
from osc_agent.tools.files import FILE_TOOLS, glob_files, read_file, write_file
from osc_agent.tools.git import GIT_TOOLS, git_status
from osc_agent.tools.repo import REPO_TOOLS, inspect_repo
from osc_agent.tools.shell import BASH_TOOL, run_bash

TEAM_ROLES = {"reviewer", "tester", "doc_writer"}
TEAMMATE_MAX_ROUNDS = 10
IDLE_POLL_INTERVAL_SECONDS = 5.0
IDLE_TIMEOUT_SECONDS = 60.0
LEAD_AGENT = "lead"

SPAWN_TEAMMATE_TOOL = {
    "name": "spawn_teammate",
    "description": "Start a long-lived teammate thread with its own inbox and limited tools.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"type": "string", "enum": sorted(TEAM_ROLES)},
            "prompt": {"type": "string"},
            "allow_write": {
                "type": "boolean",
                "description": "Grant write_file to the teammate only when the assignment explicitly requires edits.",
            },
        },
        "required": ["name", "role", "prompt"],
        "additionalProperties": False,
    },
}

SEND_MESSAGE_TOOL = {
    "name": "send_message",
    "description": "Send a message to the lead or a teammate inbox.",
    "input_schema": {
        "type": "object",
        "properties": {
            "to_agent": {"type": "string"},
            "content": {"type": "string"},
            "message_type": {"type": "string", "default": "message"},
            "metadata": {"type": "object"},
        },
        "required": ["to_agent", "content"],
    },
}

CHECK_INBOX_TOOL = {
    "name": "check_inbox",
    "description": "Read and consume messages from the lead inbox.",
    "input_schema": {"type": "object", "properties": {}},
}

SUBMIT_PLAN_TOOL = {
    "name": "request_plan_review",
    "description": "Submit a plan to lead for approval before continuing risky teammate work.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sender": {"type": "string"},
            "plan": {"type": "string"},
        },
        "required": ["sender", "plan"],
    },
}

TEAM_TOOLS = [SPAWN_TEAMMATE_TOOL, SEND_MESSAGE_TOOL, CHECK_INBOX_TOOL]
TEAMMATE_TOOLS = [
    BASH_TOOL,
    FILE_TOOLS[0],
    FILE_TOOLS[3],
    GIT_TOOLS[0],
    REPO_TOOLS[0],
    SEND_MESSAGE_TOOL,
    SUBMIT_PLAN_TOOL,
    CONTRIBUTION_TASK_TOOLS[1],
    CONTRIBUTION_TASK_TOOLS[3],
    CONTRIBUTION_TASK_TOOLS[4],
]

_bus_lock = threading.Lock()
_active_teammates: dict[str, threading.Thread] = {}


@dataclass
class TeamMessage:
    from_agent: str
    to_agent: str
    content: str
    type: str
    ts: float
    metadata: dict[str, Any]


class MessageBus:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.mailbox_dir = repo_root / ".osc_agent" / "mailboxes"

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        message_type: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """发送消息就是向目标邮箱追加一行 JSONL，便于跨线程观察。"""
        if not _valid_agent_name(to_agent):
            return f"Error: invalid agent name {to_agent}"
        message = TeamMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            type=message_type or "message",
            ts=time.time(),
            metadata=metadata or {},
        )
        self.mailbox_dir.mkdir(parents=True, exist_ok=True)
        with _bus_lock:
            with self._inbox_path(to_agent).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(message), ensure_ascii=False, default=str) + "\n")
        append_trace(self.repo_root, "team_message_sent", {"from": from_agent, "to": to_agent, "type": message.type})
        return f"Sent message to {to_agent}"

    def read_inbox(self, agent: str) -> list[dict[str, Any]]:
        """消费式读取邮箱；读完即清空，避免同一条消息反复进入上下文。"""
        path = self._inbox_path(agent)
        if not path.exists():
            return []
        with _bus_lock:
            if not path.exists():
                return []
            lines = path.read_text(encoding="utf-8").splitlines()
            path.unlink()
        messages: list[dict[str, Any]] = []
        for line in lines:
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if messages:
            append_trace(self.repo_root, "team_inbox_read", {"agent": agent, "count": len(messages)})
        return messages

    def _inbox_path(self, agent: str) -> Path:
        return self.mailbox_dir / f"{agent}.jsonl"


def spawn_teammate(
    *,
    name: str,
    role: str,
    prompt: str,
    repo_root: Path,
    client: Any,
    settings: Settings,
    allow_write: bool = False,
    autonomous: bool = True,
) -> str:
    """启动队友线程；S17 默认开启空闲轮询和自动认领。"""
    name = name.strip()
    if not _valid_agent_name(name):
        return "Error: teammate name must contain only letters, numbers, underscore, dot, or dash"
    if role not in TEAM_ROLES:
        return f"Error: role must be one of: {', '.join(sorted(TEAM_ROLES))}"
    if not prompt.strip():
        return "Error: prompt is required"

    key = f"{repo_root.resolve()}::{name}"
    if key in _active_teammates and _active_teammates[key].is_alive():
        return f"Error: teammate {name} is already active"

    bus = MessageBus(repo_root)

    def worker() -> None:
        summary = _run_teammate_loop(
            name=name,
            role=role,
            prompt=prompt,
            repo_root=repo_root,
            client=client,
            settings=settings,
            allow_write=allow_write,
            autonomous=autonomous,
        )
        bus.send(name, LEAD_AGENT, summary, "result", {"role": role})

    thread = threading.Thread(target=worker, daemon=True, name=f"teammate-{name}")
    _active_teammates[key] = thread
    thread.start()
    append_trace(repo_root, "teammate_spawned", {"name": name, "role": role, "allow_write": allow_write})
    return f"Spawned teammate {name} as {role}"


def send_message(
    *,
    repo_root: Path,
    from_agent: str = LEAD_AGENT,
    to_agent: str,
    content: str,
    message_type: str = "message",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Lead 和队友共享的发送入口；metadata 供协议阶段关联 request_id。"""
    if not content.strip():
        return "Error: content is required"
    return MessageBus(repo_root).send(from_agent, to_agent, content, message_type, metadata)


def check_inbox(*, repo_root: Path, agent: str = LEAD_AGENT) -> str:
    from osc_agent.harness.protocols import consume_inbox

    messages = consume_inbox(repo_root, agent)
    return _format_inbox(messages) if messages else "(inbox empty)"


def collect_team_notifications(repo_root: Path, *, agent: str = LEAD_AGENT) -> list[str]:
    """每轮主循环自动收取 Lead 邮箱，先路由协议，再注入文本块。"""
    from osc_agent.harness.protocols import consume_inbox

    messages = consume_inbox(repo_root, agent)
    if not messages:
        return []
    return [_format_message(message) for message in messages]


def idle_poll(
    *,
    repo_root: Path,
    name: str,
    messages: list[dict[str, Any]],
    worktree_context: dict[str, Path | None] | None = None,
) -> str:
    """队友空闲时优先处理 inbox，其次扫描任务板并自动认领可开始任务。"""
    from osc_agent.harness.protocols import consume_inbox

    deadline = time.monotonic() + IDLE_TIMEOUT_SECONDS
    while time.monotonic() <= deadline:
        time.sleep(IDLE_POLL_INTERVAL_SECONDS)
        inbox = consume_inbox(repo_root, name)
        if any(message.get("type") == "shutdown_request" for message in inbox):
            return "shutdown"
        if inbox:
            messages.append({"role": "user", "content": _format_inbox(inbox)})
            return "work"

        claimed = claim_next_available_task(repo_root=repo_root, owner=name, worktree_context=worktree_context)
        if claimed:
            messages.append({"role": "user", "content": claimed})
            return "work"
    return "timeout"


def scan_unclaimed_tasks(repo_root: Path) -> list[dict[str, Any]]:
    """扫描 pending、无 owner、依赖已完成的任务，供空闲队友自动认领。"""
    candidates = []
    for task in load_all_tasks(repo_root):
        if task.status != "pending" or task.owner:
            continue
        if blocking_dependencies(repo_root, task):
            continue
        candidates.append(task)
    return [
        {
            "id": task.id,
            "subject": task.subject,
            "description": task.description,
            "blockedBy": task.blockedBy,
            "files": task.files,
        }
        for task in candidates
    ]


def claim_next_available_task(
    *,
    repo_root: Path,
    owner: str,
    worktree_context: dict[str, Path | None] | None = None,
) -> str:
    """按稳定顺序认领第一个可开始任务；失败时继续尝试下一个候选。"""
    for task in scan_unclaimed_tasks(repo_root):
        result = claim_task(repo_root=repo_root, task_id=task["id"], owner=owner)
        if result.startswith("Claimed"):
            if worktree_context is not None:
                worktree_context["path"] = resolve_task_worktree(repo_root, task["id"])
            append_trace(repo_root, "teammate_task_claimed", {"owner": owner, "task_id": task["id"]})
            return f"[Auto-claimed task]\n{get_task(repo_root=repo_root, task_id=task['id'])}"
    return ""


def _run_teammate_loop(
    *,
    name: str,
    role: str,
    prompt: str,
    repo_root: Path,
    client: Any,
    settings: Settings,
    allow_write: bool,
    autonomous: bool,
) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tools = list(TEAMMATE_TOOLS)
    if allow_write:
        tools.append(FILE_TOOLS[1])
    worktree_context: dict[str, Path | None] = {"path": None}
    handlers = _teammate_handlers(repo_root, name=name, allow_write=allow_write, worktree_context=worktree_context)
    hook_registry = default_hooks()
    hook_context = HookContext(repo_root=repo_root)
    system_prompt = (
        f"You are teammate '{name}', a {role}. Work independently, communicate via send_message, "
        "and send a concise final result to lead when done. Do not spawn other teammates."
    )

    final_summary = ""
    try:
        while True:
            _ensure_identity(messages, name=name, role=role)
            should_idle = False

            for _round_index in range(1, TEAMMATE_MAX_ROUNDS + 1):
                from osc_agent.harness.protocols import consume_inbox

                inbox = consume_inbox(repo_root, name)
                if any(message.get("type") == "shutdown_request" for message in inbox):
                    final_summary = f"Teammate {name} shut down gracefully."
                    should_idle = False
                    break
                if inbox:
                    messages.append({"role": "user", "content": _format_inbox(inbox)})

                response = client.messages.create(
                    model=settings.model_id,
                    system=system_prompt,
                    messages=messages[-20:],
                    tools=tools,
                    max_tokens=4000,
                )
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    final_summary = _extract_text(response.content)
                    should_idle = autonomous
                    break

                results: list[dict[str, str]] = []
                waiting_for_protocol = False
                for block in response.content:
                    if _block_attr(block, "type") != "tool_use":
                        continue
                    tool_name = _block_attr(block, "name")
                    tool_args = _tool_input(block)
                    handler = handlers.get(tool_name)
                    started = perf_counter()
                    pre_results = hook_registry.run(
                        "PreToolUse",
                        hook_context,
                        {"tool_name": tool_name, "tool_args": tool_args},
                    )
                    blocked = next((result for result in pre_results if not result.allowed), None)
                    if blocked is not None:
                        output = blocked.content or "Permission denied"
                    elif handler is None:
                        output = f"Error: unknown teammate tool {tool_name}"
                    else:
                        try:
                            output = handler(**tool_args)
                        except (TypeError, ValueError) as exc:
                            output = f"Error: invalid arguments for {tool_name}: {exc}"
                    hook_registry.run(
                        "PostToolUse",
                        hook_context,
                        {
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "output": output,
                            "latency_ms": elapsed_ms(started),
                        },
                    )
                    append_trace(
                        repo_root,
                        "teammate_tool_use",
                        {"name": name, "role": role, "tool": tool_name, "output_preview": preview(output)},
                    )
                    results.append({"type": "tool_result", "tool_use_id": _block_attr(block, "id"), "content": output})
                    if tool_name == "request_plan_review":
                        # 计划审批是执行门：提交计划后先停下来，等待 Lead 明确审批。
                        waiting_for_protocol = True
                messages.append({"role": "user", "content": results})
                if waiting_for_protocol:
                    final_summary = f"Teammate {name} is waiting for plan approval."
                    should_idle = False
                    break
            else:
                final_summary = f"Teammate {name} stopped after {TEAMMATE_MAX_ROUNDS} rounds without a final answer."
                should_idle = autonomous

            if not should_idle:
                break

            idle_result = idle_poll(
                repo_root=repo_root,
                name=name,
                messages=messages,
                worktree_context=worktree_context,
            )
            if idle_result == "work":
                continue
            if idle_result == "timeout":
                final_summary = f"{final_summary}\nIdle timeout: no inbox messages or claimable tasks."
            else:
                final_summary = f"Teammate {name} shut down gracefully."
            break
    except Exception as exc:  # pragma: no cover - protects the main CLI from teammate thread failures
        final_summary = f"Error: teammate {name} failed: {exc}"

    append_trace(repo_root, "teammate_finished", {"name": name, "role": role, "summary": preview(final_summary)})
    return f"Teammate {name} ({role}) result:\n{final_summary}".strip()


def _teammate_handlers(
    repo_root: Path,
    *,
    name: str,
    allow_write: bool,
    worktree_context: dict[str, Path | None] | None = None,
) -> dict[str, Any]:
    def cwd() -> Path:
        return (worktree_context or {}).get("path") or repo_root

    def claim_for_teammate(task_id: str, owner: str = name) -> str:
        result = claim_task(repo_root=repo_root, task_id=task_id, owner=owner)
        if result.startswith("Claimed") and worktree_context is not None:
            # 认领绑定 worktree 的任务后，后续工具自动在隔离目录执行。
            worktree_context["path"] = resolve_task_worktree(repo_root, task_id)
        return result

    handlers: dict[str, Any] = {
        "bash": lambda command, run_in_background=False: _run_read_only_bash(command, repo_root=cwd()),
        "read_file": lambda path, limit=20_000, offset=0: read_file(
            repo_root=cwd(),
            path=path,
            limit=limit,
            offset=offset,
        ),
        "glob": lambda pattern: glob_files(repo_root=cwd(), pattern=pattern),
        "git_status": lambda: git_status(repo_root=cwd()),
        "inspect_repo": lambda: inspect_repo(repo_root=cwd()),
        "send_message": lambda to_agent, content, message_type="message", metadata=None: send_message(
            repo_root=repo_root,
            from_agent=name,
            to_agent=to_agent,
            content=content,
            message_type=message_type,
            metadata=metadata,
        ),
    }
    from osc_agent.harness.protocols import request_plan_review

    handlers["request_plan_review"] = lambda sender=name, plan="": request_plan_review(
        repo_root=repo_root,
        sender=sender,
        plan=plan,
    )
    handlers["list_tasks"] = lambda: list_tasks(repo_root=repo_root)
    handlers["claim_task"] = claim_for_teammate
    handlers["complete_task"] = lambda task_id, evidence=None: complete_task(
        repo_root=repo_root,
        task_id=task_id,
        evidence=evidence,
    )
    if allow_write:
        handlers["write_file"] = lambda path, content: write_file(
            repo_root=cwd(),
            path=path,
            content=content,
            enforce_permissions=True,
        )
    return handlers


def _run_read_only_bash(command: str, *, repo_root: Path) -> str:
    normalized = command.strip().lower()
    if not any(normalized == prefix.strip() or normalized.startswith(prefix) for prefix in READ_ONLY_BASH_PREFIXES):
        return "Permission denied: teammate bash is read-only"
    return run_bash(command, repo_root=repo_root, enforce_permissions=True)


def _ensure_identity(messages: list[dict[str, Any]], *, name: str, role: str) -> None:
    identity = f"<identity>You are teammate '{name}', role: {role}. Continue autonomous team work.</identity>"
    if not messages or messages[0].get("content") != identity:
        messages.insert(0, {"role": "user", "content": identity})


def _format_inbox(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_format_message(message) for message in messages)


def _format_message(message: dict[str, Any]) -> str:
    return (
        "<teammate-message>\n"
        f"  <from>{message.get('from_agent')}</from>\n"
        f"  <to>{message.get('to_agent')}</to>\n"
        f"  <type>{message.get('type')}</type>\n"
        f"  <content>{message.get('content')}</content>\n"
        "</teammate-message>"
    )


def _valid_agent_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name))


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _tool_input(block: Any) -> dict[str, Any]:
    value = _block_attr(block, "input", {})
    return value if isinstance(value, dict) else {}


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = _block_attr(block, "text")
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return str(content)
