"""
Lead Agent 初始只有内置工具 + connect_mcp
    ↓
LLM 调用 connect_mcp(server_name)
    ↓
连接 mock MCP server，并发现 server 提供的工具
    ↓
把 MCP client 保存到 _mcp_clients
    ↓
下一轮 agent_loop 重新 assemble_tool_pool()
    ↓
MCP 工具以 mcp__server__tool 的名字加入工具池
    ↓
LLM 可以调用这些动态工具
    ↓
assemble_tool_handlers() 把调用转发给对应 MCPClient.call_tool()
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from typing import Any

CONNECT_MCP_TOOL = {
    "name": "connect_mcp",
    "description": "Connect to a mock MCP server and discover its tools.",
    "input_schema": {
        "type": "object",
        "properties": {
            "server_name": {
                "type": "string",
                "description": "Mock MCP server name, for example docs, deploy, or issues.",
            }
        },
        "required": ["server_name"],
    },
}

_DISALLOWED_MCP_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_mcp_clients: dict[str, "MCPClient"] = {}


class MCPClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: list[dict[str, Any]] = []
        self._handlers: dict[str, Callable[..., str]] = {}

    def register(self, tool_defs: list[dict[str, Any]], handlers: dict[str, Callable[..., str]]) -> None:
        """模拟 MCP tools/list 发现结果，并保存 tools/call 的处理函数。"""
        self.tools = tool_defs
        self._handlers = handlers

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """模拟 MCP tools/call；真实实现会走 JSON-RPC transport。"""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return handler(**args)
        except TypeError as exc:
            return f"MCP error: invalid arguments for '{tool_name}': {exc}"


def normalize_mcp_name(name: str) -> str:
    """规范化 server/tool 名，避免特殊字符造成工具名冲突或注入。"""
    normalized = _DISALLOWED_MCP_CHARS.sub("_", name.strip())
    return normalized or "unnamed"


def prefixed_mcp_tool_name(server_name: str, tool_name: str) -> str:
    return f"mcp__{normalize_mcp_name(server_name)}__{normalize_mcp_name(tool_name)}"


def connect_mcp(server_name: str) -> str:
    name = server_name.strip()
    if name in _mcp_clients:
        return f"MCP server '{name}' already connected"
    factory = MOCK_SERVERS.get(name)
    if factory is None:
        return "Unknown MCP server '{}'. Available: {}".format(name, ", ".join(sorted(MOCK_SERVERS)))
    client = factory()
    _mcp_clients[name] = client
    discovered = ", ".join(prefixed_mcp_tool_name(name, tool["name"]) for tool in client.tools)
    return f"Connected to '{name}'. Discovered tools: {discovered}"


def assemble_tool_pool(builtin_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每轮重新组装工具池；connect_mcp 后新工具才能进入下一轮 prompt。"""
    tools = [copy.deepcopy(tool) for tool in builtin_tools]
    for server_name, client in _mcp_clients.items():
        for tool_def in client.tools:
            tools.append(
                {
                    "name": prefixed_mcp_tool_name(server_name, tool_def["name"]),
                    "description": _annotated_description(tool_def),
                    "input_schema": copy.deepcopy(tool_def.get("input_schema", {"type": "object", "properties": {}})),
                }
            )
    return tools


def assemble_tool_handlers(base_handlers: dict[str, Any]) -> dict[str, Any]:
    handlers = dict(base_handlers)
    for server_name, client in _mcp_clients.items():
        for tool_def in client.tools:
            original_name = tool_def["name"]
            exposed_name = prefixed_mcp_tool_name(server_name, original_name)
            handlers[exposed_name] = (
                lambda _client=client, _tool=original_name, **kwargs: _client.call_tool(_tool, kwargs)
            )
    return handlers


def reset_mcp_clients() -> None:
    """测试用：清空已连接 server，避免跨测试污染动态工具池。"""
    _mcp_clients.clear()


def _annotated_description(tool_def: dict[str, Any]) -> str:
    description = str(tool_def.get("description", "MCP tool"))
    annotations = tool_def.get("annotations") if isinstance(tool_def.get("annotations"), dict) else {}
    if annotations.get("destructive"):
        return f"{description} (destructive)"
    if annotations.get("readOnly"):
        return f"{description} (readOnly)"
    return description


def _docs_server() -> MCPClient:
    client = MCPClient("docs")
    client.register(
        [
            {
                "name": "search",
                "description": "Search project documentation.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "annotations": {"readOnly": True},
            },
            {
                "name": "read_page",
                "description": "Read one documentation page.",
                "input_schema": {
                    "type": "object",
                    "properties": {"page_id": {"type": "string"}},
                    "required": ["page_id"],
                },
                "annotations": {"readOnly": True},
            },
        ],
        {
            "search": lambda query: f"docs search results for: {query}",
            "read_page": lambda page_id: f"docs page {page_id}: contribution guide",
        },
    )
    return client


def _deploy_server() -> MCPClient:
    client = MCPClient("deploy")
    client.register(
        [
            {
                "name": "trigger",
                "description": "Trigger a deployment.",
                "input_schema": {
                    "type": "object",
                    "properties": {"environment": {"type": "string"}},
                    "required": ["environment"],
                },
                "annotations": {"destructive": True},
            },
            {
                "name": "logs",
                "description": "Read deployment logs.",
                "input_schema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
                "annotations": {"readOnly": True},
            },
        ],
        {
            "trigger": lambda environment: f"deployment triggered for {environment}",
            "logs": lambda service: f"logs for {service}: ok",
        },
    )
    return client


def _issues_server() -> MCPClient:
    client = MCPClient("issues")
    client.register(
        [
            {
                "name": "lookup",
                "description": "Lookup an issue.",
                "input_schema": {
                    "type": "object",
                    "properties": {"issue_id": {"type": "string"}},
                    "required": ["issue_id"],
                },
                "annotations": {"readOnly": True},
            }
        ],
        {"lookup": lambda issue_id: f"issue {issue_id}: open"},
    )
    return client


MOCK_SERVERS: dict[str, Callable[[], MCPClient]] = {
    "docs": _docs_server,
    "deploy": _deploy_server,
    "issues": _issues_server,
}
