"""
某个 agent 发起协议请求
    ↓
创建 ProtocolState（request_id / type / sender / target / status）
    ↓
写入 .osc_agent/protocols.json 持久化
    ↓
通过 MessageBus 把请求消息发给目标 agent
    ↓
目标 agent 读取 inbox
    ↓
route_protocol_message 根据 request_id / message_type 路由协议消息
    ↓
目标 agent 返回 response 消息
    ↓
发送方读取 response
    ↓
match_response 校验 request_id + response_type
    ↓
更新 ProtocolState 为 approved / rejected
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any

from osc_agent.harness.trace import append_trace

PROTOCOL_TYPES = {"shutdown", "plan_approval", "write_approval"}
PROTOCOL_RESPONSE_TYPES = {
    "shutdown": "shutdown_response",
    "plan_approval": "plan_approval_response",
    "write_approval": "write_approval_response",
}
_protocol_lock = threading.RLock()

PROTOCOL_TOOLS = [
    {
        "name": "request_shutdown",
        "description": "Ask a teammate to shut down gracefully using a request/response handshake.",
        "input_schema": {
            "type": "object",
            "properties": {"target": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["target"],
        },
    },
    {
        "name": "request_plan_review",
        "description": "Submit a teammate plan to lead for approval before continuing risky work.",
        "input_schema": {
            "type": "object",
            "properties": {"sender": {"type": "string"}, "plan": {"type": "string"}},
            "required": ["sender", "plan"],
        },
    },
    {
        "name": "review_plan",
        "description": "Approve or reject a pending plan approval request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "feedback": {"type": "string"},
            },
            "required": ["request_id", "approve"],
        },
    },
    {
        "name": "request_write_approval",
        "description": "Submit a write operation request to lead for approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sender": {"type": "string"},
                "path": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["sender", "path", "reason"],
        },
    },
]


@dataclass
class ProtocolState:
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: dict[str, Any]
    created_at: float
    resolved_at: float | None = None


def request_shutdown(*, repo_root: Path, target: str, reason: str = "") -> str:
    """Lead 请求队友体面关机；队友收到后要回 shutdown_response。"""
    state = _create_state(
        repo_root=repo_root,
        protocol_type="shutdown",
        sender="lead",
        target=target,
        payload={"reason": reason},
    )
    _send_protocol_message(repo_root, state, "shutdown_request", reason or "Please shut down gracefully.")
    return _state_json(state)


def request_plan_review(*, repo_root: Path, sender: str, plan: str) -> str:
    """队友提交计划审批；审批完成前应等待 lead 的 plan_approval_response。"""
    if not plan.strip():
        return "Error: plan is required"
    state = _create_state(
        repo_root=repo_root,
        protocol_type="plan_approval",
        sender=sender,
        target="lead",
        payload={"plan": plan},
    )
    _send_protocol_message(repo_root, state, "plan_approval_request", plan)
    return _state_json(state)


def request_write_approval(*, repo_root: Path, sender: str, path: str, reason: str) -> str:
    """队友请求写权限；S16 只实现握手记录，不直接放开写操作。"""
    if not path.strip() or not reason.strip():
        return "Error: path and reason are required"
    state = _create_state(
        repo_root=repo_root,
        protocol_type="write_approval",
        sender=sender,
        target="lead",
        payload={"path": path, "reason": reason},
    )
    _send_protocol_message(repo_root, state, "write_approval_request", f"{path}: {reason}")
    return _state_json(state)


def review_plan(*, repo_root: Path, request_id: str, approve: bool, feedback: str = "") -> str:
    """Lead 审批计划；只有 plan_approval 类型的 pending 请求能被该函数处理。"""
    state = load_protocol_state(repo_root, request_id)
    if state is None:
        return f"Error: unknown request {request_id}"
    if state.type != "plan_approval":
        return f"Error: request {request_id} is {state.type}, not plan_approval"
    result = _resolve_state(repo_root, request_id, approve=approve, expected_type="plan_approval")
    if result.startswith("Error:"):
        return result
    _send_response(repo_root, state, "plan_approval_response", approve, feedback)
    return result


def match_response(
    *,
    repo_root: Path,
    response_type: str,
    request_id: str,
    approve: bool,
    responder: str | None = None,
    recipient: str | None = None,
) -> str:
    """按 request_id 和 response_type 匹配响应，防止错误响应更新错误协议。"""
    state = load_protocol_state(repo_root, request_id)
    if state is None:
        return f"Error: unknown request {request_id}"
    expected = PROTOCOL_RESPONSE_TYPES.get(state.type)
    if response_type != expected:
        return f"Error: response type {response_type} does not match {state.type}"
    if responder is not None and responder != state.target:
        return f"Error: response sender {responder} does not match {state.target}"
    if recipient is not None and recipient != state.sender:
        return f"Error: response recipient {recipient} does not match {state.sender}"
    return _resolve_state(repo_root, request_id, approve=approve, expected_type=state.type)


def consume_inbox(repo_root: Path, agent: str) -> list[dict[str, Any]]:
    """统一消费 inbox：先路由协议消息，再把原消息返回给调用方注入上下文。"""
    from osc_agent.harness.teams import MessageBus

    messages = MessageBus(repo_root).read_inbox(agent)
    for message in messages:
        route_protocol_message(repo_root=repo_root, agent=agent, message=message)
    return messages


def route_protocol_message(*, repo_root: Path, agent: str, message: dict[str, Any]) -> str | None:
    message_type = str(message.get("type", "message"))
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    request_id = str(metadata.get("request_id", ""))
    if request_id and message_type.endswith("_response"):
        return match_response(
            repo_root=repo_root,
            response_type=message_type,
            request_id=request_id,
            approve=bool(metadata.get("approve", False)),
            responder=str(message.get("from_agent", "")),
            recipient=agent,
        )
    if message_type == "shutdown_request" and request_id:
        state = load_protocol_state(repo_root, request_id)
        if (
            state is None
            or state.type != "shutdown"
            or state.sender != str(message.get("from_agent", ""))
            or state.target != agent
        ):
            return "Error: shutdown request does not match a pending protocol"
        _send_response(repo_root, _message_state(message, request_id), "shutdown_response", True, "shutdown accepted")
        append_trace(repo_root, "protocol_shutdown_ack", {"agent": agent, "request_id": request_id})
        return "shutdown"
    return None


def load_protocol_state(repo_root: Path, request_id: str) -> ProtocolState | None:
    with _protocol_lock:
        for state in _load_states(repo_root):
            if state.request_id == request_id:
                return state
    return None


def protocols_path(repo_root: Path) -> Path:
    return repo_root / ".osc_agent" / "protocols.json"


def _create_state(
    *,
    repo_root: Path,
    protocol_type: str,
    sender: str,
    target: str,
    payload: dict[str, Any],
) -> ProtocolState:
    if protocol_type not in PROTOCOL_TYPES:
        raise ValueError(f"unknown protocol type: {protocol_type}")
    state = ProtocolState(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        type=protocol_type,
        sender=sender,
        target=target,
        status="pending",
        payload=payload,
        created_at=time(),
    )
    with _protocol_lock:
        states = _load_states(repo_root)
        states.append(state)
        _save_states(repo_root, states)
    append_trace(repo_root, "protocol_request", {"request_id": state.request_id, "type": state.type})
    return state


def _resolve_state(repo_root: Path, request_id: str, *, approve: bool, expected_type: str) -> str:
    with _protocol_lock:
        states = _load_states(repo_root)
        for state in states:
            if state.request_id != request_id:
                continue
            if state.type != expected_type:
                return f"Error: request {request_id} is {state.type}, not {expected_type}"
            if state.status != "pending":
                return f"Error: request {request_id} is already {state.status}"
            state.status = "approved" if approve else "rejected"
            state.resolved_at = time()
            _save_states(repo_root, states)
            append_trace(repo_root, "protocol_response", {"request_id": request_id, "status": state.status})
            return f"Request {request_id} {state.status}"
        return f"Error: unknown request {request_id}"


def _send_protocol_message(repo_root: Path, state: ProtocolState, message_type: str, content: str) -> None:
    from osc_agent.harness.teams import MessageBus

    MessageBus(repo_root).send(
        state.sender,
        state.target,
        content,
        message_type,
        {"request_id": state.request_id, "protocol_type": state.type},
    )


def _send_response(repo_root: Path, state: ProtocolState, response_type: str, approve: bool, feedback: str = "") -> None:
    from osc_agent.harness.teams import MessageBus

    MessageBus(repo_root).send(
        state.target,
        state.sender,
        feedback or ("approved" if approve else "rejected"),
        response_type,
        {"request_id": state.request_id, "approve": approve, "protocol_type": state.type},
    )


def _message_state(message: dict[str, Any], request_id: str) -> ProtocolState:
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    return ProtocolState(
        request_id=request_id,
        type=str(metadata.get("protocol_type", "shutdown")),
        sender=str(message.get("from_agent", "")),
        target=str(message.get("to_agent", "")),
        status="pending",
        payload={"content": message.get("content", "")},
        created_at=float(message.get("ts", time())),
    )


def _load_states(repo_root: Path) -> list[ProtocolState]:
    with _protocol_lock:
        path = protocols_path(repo_root)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            ProtocolState(
                request_id=str(item.get("request_id", "")),
                type=str(item.get("type", "")),
                sender=str(item.get("sender", "")),
                target=str(item.get("target", "")),
                status=str(item.get("status", "pending")),
                payload=item.get("payload") if isinstance(item.get("payload"), dict) else {},
                created_at=float(item.get("created_at", 0)),
                resolved_at=item.get("resolved_at"),
            )
            for item in data.get("requests", [])
        ]


def _save_states(repo_root: Path, states: list[ProtocolState]) -> None:
    with _protocol_lock:
        path = protocols_path(repo_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            temp.write_text(
                json.dumps({"requests": [asdict(state) for state in states]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()


def _state_json(state: ProtocolState) -> str:
    return json.dumps(asdict(state), ensure_ascii=False, indent=2)
