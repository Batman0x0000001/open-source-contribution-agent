from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from osc_agent.agent_loop import TOOLS, agent_loop, build_tool_handlers
from osc_agent.config import Settings
from osc_agent.harness.mcp import (
    assemble_tool_handlers,
    assemble_tool_pool,
    connect_mcp,
    normalize_mcp_name,
    prefixed_mcp_tool_name,
    reset_mcp_clients,
)


@pytest.fixture(autouse=True)
def _reset_mcp():
    reset_mcp_clients()
    yield
    reset_mcp_clients()


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


def test_normalize_mcp_name_replaces_unsafe_characters():
    assert normalize_mcp_name("docs server/v1") == "docs_server_v1"
    assert normalize_mcp_name("") == "unnamed"
    assert prefixed_mcp_tool_name("docs server", "search/docs") == "mcp__docs_server__search_docs"


def test_connect_mcp_discovers_prefixed_tools():
    result = connect_mcp("docs")
    tools = assemble_tool_pool([])

    assert "Connected to 'docs'" in result
    assert "mcp__docs__search" in result
    assert {tool["name"] for tool in tools} == {"mcp__docs__search", "mcp__docs__read_page"}
    assert all("(readOnly)" in tool["description"] for tool in tools)


def test_connect_mcp_rejects_unknown_server():
    result = connect_mcp("missing")

    assert result.startswith("Unknown MCP server 'missing'")
    assert "docs" in result


def test_assemble_tool_handlers_calls_mcp_tool():
    connect_mcp("docs")
    handlers = assemble_tool_handlers({"builtin": lambda: "ok"})

    assert handlers["builtin"]() == "ok"
    assert handlers["mcp__docs__search"](query="hooks") == "docs search results for: hooks"


def test_deploy_tool_is_marked_destructive():
    connect_mcp("deploy")
    tools = assemble_tool_pool([])
    descriptions = {tool["name"]: tool["description"] for tool in tools}

    assert "(destructive)" in descriptions["mcp__deploy__trigger"]
    assert "(readOnly)" in descriptions["mcp__deploy__logs"]


def test_agent_loop_registers_connect_mcp_tool(tmp_path):
    tool_names = {tool["name"] for tool in TOOLS}
    handlers = build_tool_handlers(tmp_path)

    assert "connect_mcp" in tool_names
    assert "mcp__issues__lookup" in handlers["connect_mcp"](server_name="issues")


class ConnectThenStopMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="connect_mcp",
                        id="toolu_connect",
                        input={"server_name": "docs"},
                    )
                ],
            )
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="done")])


class FakeClient:
    def __init__(self, messages) -> None:
        self.messages = messages


def test_agent_loop_rebuilds_tool_pool_after_connect_mcp(tmp_path):
    fake_messages = ConnectThenStopMessages()
    messages = [{"role": "user", "content": "connect docs"}]

    agent_loop(
        messages,
        client=FakeClient(fake_messages),
        settings=_settings(),
        repo_root=tmp_path,
    )

    first_tools = {tool["name"] for tool in fake_messages.calls[0]["tools"]}
    second_tools = {tool["name"] for tool in fake_messages.calls[1]["tools"]}

    assert "mcp__docs__search" not in first_tools
    assert "mcp__docs__search" in second_tools
    assert "mcp__docs__search" not in fake_messages.calls[0]["system"]
    assert "mcp__docs__search" in fake_messages.calls[1]["system"]


def test_mcp_connections_are_isolated_by_session():
    connect_mcp("docs", session_id="one")

    assert {tool["name"] for tool in assemble_tool_pool([], session_id="one")}
    assert assemble_tool_pool([], session_id="two") == []
