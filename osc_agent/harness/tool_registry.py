from __future__ import annotations

from typing import Any, Iterable

from osc_agent.harness.mcp import assemble_tool_handlers, assemble_tool_pool


class ToolRegistry:
    """统一保存工具 Schema、绑定后的 Handler 和副作用元数据。"""

    def __init__(
        self,
        schemas: Iterable[dict[str, Any]],
        handlers: dict[str, Any],
        *,
        side_effect_tools: set[str] | None = None,
        require_complete_handlers: bool = False,
    ) -> None:
        self._schemas: list[dict[str, Any]] = []
        self._schema_names: set[str] = set()
        for schema in schemas:
            name = str(schema.get("name", ""))
            if not name:
                raise ValueError("tool schema requires a non-empty name")
            if name in self._schema_names:
                raise ValueError(f"duplicate tool schema: {name}")
            self._schema_names.add(name)
            self._schemas.append(schema)

        effects = set(side_effect_tools or ())
        unknown_effects = effects - self._schema_names
        if unknown_effects:
            names = ", ".join(sorted(unknown_effects))
            raise ValueError(f"side-effect metadata references unknown tools: {names}")
        self._handlers = dict(handlers)
        if require_complete_handlers:
            missing_handlers = self._schema_names - self._handlers.keys()
            unknown_handlers = self._handlers.keys() - self._schema_names
            if missing_handlers or unknown_handlers:
                missing = ", ".join(sorted(missing_handlers)) or "none"
                unknown = ", ".join(sorted(unknown_handlers)) or "none"
                raise ValueError(
                    f"tool registry handler mismatch; missing: {missing}; unknown: {unknown}"
                )
        self._side_effect_tools = effects

    def schemas(self, *, session_id: str = "default") -> list[dict[str, Any]]:
        return assemble_tool_pool(self._schemas, session_id=session_id)

    def handlers(self, *, session_id: str = "default") -> dict[str, Any]:
        return assemble_tool_handlers(self._handlers, session_id=session_id)

    def has_side_effect(self, tool_name: str) -> bool:
        return tool_name in self._side_effect_tools
