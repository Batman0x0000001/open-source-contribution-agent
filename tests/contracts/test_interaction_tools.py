from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.models import Ask, ToolUseBlock, ToolUseContext
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor
from tests.contracts.registry_factory import build_test_tool_registry as build_core_tool_registry


def context(root: Path, *, mode: str = "default", plan_path: str | None = None) -> ToolUseContext:
    return ToolUseContext(
        session_id="session-1",
        working_directory=str(root),
        repository_root=str(root),
        state_directory=str(root / "state"),
        permission_mode=mode,
        plan_path=plan_path,
    )


def test_ask_user_question_returns_answers_to_agent_loop(tmp_path: Path) -> None:
    async def answer(questions):
        return {questions[0]["question"]: "Small fix"}

    executor = ToolExecutor(build_core_tool_registry(), dependencies=ToolExecutionDependencies(question_handler=answer))
    result = asyncio.run(executor.execute(ToolUseBlock(id="q", name="ask_user_question", input={"questions": [{"header": "Scope", "question": "Choose?", "options": [{"label": "Small fix", "description": "Low risk"}, {"label": "Feature", "description": "More work"}]}]}), context(tmp_path)))

    assert result.error is None
    assert result.data["answers"] == {"Choose?": "Small fix"}


def test_plan_mode_blocks_writes_but_allows_fixed_plan_file(tmp_path: Path) -> None:
    async def approve(decision: Ask) -> bool:
        return True

    executor = ToolExecutor(build_core_tool_registry(), dependencies=ToolExecutionDependencies(approval_handler=approve))
    blocked = asyncio.run(executor.execute(ToolUseBlock(id="w", name="write_file", input={"path": "x.txt", "content": "x"}), context(tmp_path, mode="plan")))
    plan = asyncio.run(executor.execute(ToolUseBlock(id="p", name="write_plan", input={"content": "# Plan"}), context(tmp_path, mode="plan")))

    assert blocked.error and blocked.error.code == "PERMISSION_DENIED"
    assert plan.error is None
    assert (tmp_path / "state" / "plans" / "session-1.md").read_text(encoding="utf-8") == "# Plan"


def test_exit_plan_mode_requires_existing_plan(tmp_path: Path) -> None:
    executor = ToolExecutor(build_core_tool_registry())
    result = asyncio.run(executor.execute(ToolUseBlock(id="exit", name="exit_plan_mode", input={}), context(tmp_path, mode="plan")))
    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"


def test_plan_cannot_be_read_through_another_session_path(tmp_path: Path) -> None:
    plans = tmp_path / "state" / "plans"
    plans.mkdir(parents=True)
    (plans / "other.md").write_text("secret", encoding="utf-8")
    executor = ToolExecutor(build_core_tool_registry())
    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="read", name="read_plan", input={}),
            context(tmp_path, plan_path="other.md"),
        )
    )
    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
