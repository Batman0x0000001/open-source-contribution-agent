from __future__ import annotations

import asyncio
from pathlib import Path

from osc_agent.runtime.models import ApprovalResponse, Ask, ToolUseBlock, ToolUseContext
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
        return {questions[0]["id"]: "small_fix"}

    executor = ToolExecutor(build_core_tool_registry(), dependencies=ToolExecutionDependencies(question_handler=answer))
    result = asyncio.run(executor.execute(ToolUseBlock(id="q", name="ask_user_question", input={"questions": [{"id": "scope", "header": "Scope", "question": "Choose?", "options": [{"id": "small_fix", "label": "Small fix", "description": "Low risk"}, {"id": "feature", "label": "Feature", "description": "More work"}]}]}), context(tmp_path)))

    assert result.error is None
    assert result.data["answers"] == [
        {
            "question_id": "scope",
            "selected_option_id": "small_fix",
            "custom_text": None,
        }
    ]


def test_plan_mode_blocks_writes_but_allows_fixed_plan_file(tmp_path: Path) -> None:
    async def approve(decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

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


def test_verification_waiver_requires_structured_risk_and_exact_option_ids(
    tmp_path: Path,
) -> None:
    async def answer(questions):
        return {questions[0]["id"]: "proceed_without_tests"}

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(question_handler=answer),
    )
    invalid = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="invalid",
                name="ask_user_question",
                input={
                    "questions": [
                        {
                            "id": "waiver",
                            "header": "Tests",
                            "question": "Proceed without tests?",
                            "purpose": "verification_waiver",
                            "options": [
                                {
                                    "id": "yes",
                                    "label": "Proceed",
                                    "description": "Continue",
                                },
                                {
                                    "id": "no",
                                    "label": "Stop",
                                    "description": "Do not continue",
                                },
                            ],
                        }
                    ]
                },
            ),
            context(tmp_path),
        )
    )

    assert invalid.error and invalid.error.code == "TOOL_INPUT_INVALID"


def test_independent_verification_waiver_requires_bound_child_and_exact_options(
    tmp_path: Path,
) -> None:
    async def answer(questions):
        return {questions[0]["id"]: "proceed_with_partial_verification"}

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(question_handler=answer),
    )
    payload = {
        "questions": [
            {
                "id": "partial",
                "header": "Verify",
                "question": "Proceed with partial verification?",
                "purpose": "independent_verification_waiver",
                "related_child_session_id": "verify-child",
                "reason": "Docker is unavailable.",
                "alternative_checks": ["Unit tests passed"],
                "unverified_checks": ["Container startup"],
                "risks": ["Container behavior is unverified"],
                "options": [
                    {
                        "id": "proceed_with_partial_verification",
                        "label": "Proceed",
                        "description": "Accept the documented risk",
                    },
                    {
                        "id": "return_to_implementation",
                        "label": "Return",
                        "description": "Do not complete",
                    },
                ],
            }
        ]
    }

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="partial", name="ask_user_question", input=payload),
            context(tmp_path),
        )
    )
    assert result.error is None
    assert result.data["answers"][0]["selected_option_id"] == "proceed_with_partial_verification"

    invalid_question = payload["questions"][0].copy()
    invalid_question.pop("related_child_session_id")
    invalid = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="invalid-partial",
                name="ask_user_question",
                input={"questions": [invalid_question]},
            ),
            context(tmp_path),
        )
    )
    assert invalid.error and invalid.error.code == "TOOL_INPUT_INVALID"


def test_remote_mismatch_question_requires_exact_repository_pair(
    tmp_path: Path,
) -> None:
    async def answer(questions):
        return {questions[0]["id"]: "proceed"}

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(question_handler=answer),
    )
    base = {
        "id": "remote",
        "header": "Remote",
        "question": "Use the requested repository?",
        "purpose": "remote_mismatch",
        "options": [
            {
                "id": "proceed",
                "label": "Proceed",
                "description": "Use this exact repository",
            },
            {
                "id": "stop",
                "label": "Stop",
                "description": "Do not use it",
            },
        ],
    }
    invalid = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="invalid-remote",
                name="ask_user_question",
                input={"questions": [base]},
            ),
            context(tmp_path),
        )
    )
    valid_question = {
        **base,
        "remote_mismatch_reference": {
            "local_origin": "local/project",
            "requested_repository": "acme/project",
        },
    }
    valid = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="valid-remote",
                name="ask_user_question",
                input={"questions": [valid_question]},
            ),
            context(tmp_path),
        )
    )

    assert invalid.error and invalid.error.code == "TOOL_INPUT_INVALID"
    assert valid.error is None
