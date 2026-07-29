"""验证状态结果工具的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio
from pathlib import Path

from osc_agent.runtime.messages import ToolUseBlock
from osc_agent.runtime.session_store import FileToolResultStore
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.tools.state import ReadToolResultTool


def context(root: Path, session_id: str) -> tool_context:
    return tool_context(
        session_id=session_id,
        working_directory=str(root),
        state_directory=str(root / "state"),
    )


def test_tool_result_is_scoped_to_owning_session(tmp_path: Path) -> None:
    store = FileToolResultStore(tmp_path / "results")
    result_id = store.persist(session_id="owner", tool_use_id="call-1", content="full result")
    executor = ToolExecutor(ToolRegistry([ReadToolResultTool(store)]))

    owned = asyncio.run(
        executor.execute(
            ToolUseBlock(id="read-1", name="read_tool_result", input={"result_id": result_id}),
            context(tmp_path, "owner"),
        )
    )
    foreign = asyncio.run(
        executor.execute(
            ToolUseBlock(id="read-2", name="read_tool_result", input={"result_id": result_id}),
            context(tmp_path, "other"),
        )
    )

    assert owned.data["content"] == "full result"
    assert foreign.error and foreign.error.code == "TOOL_EXECUTION_FAILED"
