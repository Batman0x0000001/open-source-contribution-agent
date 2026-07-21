from __future__ import annotations

import pytest

from osc_agent.harness.tool_registry import ToolRegistry


SCHEMA = {
    "name": "read_demo",
    "description": "Read demo data.",
    "input_schema": {"type": "object", "properties": {}},
}


def test_tool_registry_keeps_schema_handler_and_side_effect_together():
    handler = lambda: "ok"
    registry = ToolRegistry([SCHEMA], {"read_demo": handler}, side_effect_tools={"read_demo"})

    assert registry.schemas()[0]["name"] == "read_demo"
    assert registry.handlers()["read_demo"] is handler
    assert registry.has_side_effect("read_demo")


def test_tool_registry_rejects_duplicate_schema_names():
    with pytest.raises(ValueError, match="duplicate tool schema"):
        ToolRegistry([SCHEMA, SCHEMA], {})


def test_tool_registry_rejects_unknown_side_effect_metadata():
    with pytest.raises(ValueError, match="side-effect metadata"):
        ToolRegistry([SCHEMA], {}, side_effect_tools={"missing"})


def test_tool_registry_can_require_complete_handler_bindings():
    with pytest.raises(ValueError, match="handler mismatch"):
        ToolRegistry([SCHEMA], {}, require_complete_handlers=True)
