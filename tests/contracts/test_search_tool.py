"""验证文本搜索工具的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.models import ToolUseBlock, ToolUseContext
from osc_agent.runtime.tool import ToolRegistry
from osc_agent.runtime.tool_execution import ToolExecutor
from osc_agent.tools.search import GrepTool


def context(root: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="grep",
        working_directory=str(root),
        repository_root=str(root),
        state_directory=str(root / "state"),
    )


def test_grep_returns_bounded_structured_matches_and_honors_ignore(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("Needle\nneedle\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Needle\n", encoding="utf-8")
    tool = GrepTool()
    executor = ToolExecutor(ToolRegistry([tool]))

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="grep",
                name="grep",
                input={
                    "pattern": "needle",
                    "case_sensitive": False,
                    "max_matches": 1,
                },
            ),
            context(tmp_path),
        )
    )

    assert result.error is None
    assert result.data["matches"] == [
        {"path": "visible.txt", "line": 1, "text": "Needle"}
    ]
    assert result.data["truncated"] is True


def test_grep_rejects_path_escape_and_option_injection(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry([GrepTool()]))

    escaped = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="escape",
                name="grep",
                input={"pattern": "x", "path": "../outside"},
            ),
            context(tmp_path),
        )
    )
    injected = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="inject",
                name="grep",
                input={"pattern": "x", "glob": "--pre=evil"},
            ),
            context(tmp_path),
        )
    )

    assert escaped.error and escaped.error.code == "TOOL_VALIDATION_FAILED"
    assert injected.error and injected.error.code == "TOOL_VALIDATION_FAILED"
