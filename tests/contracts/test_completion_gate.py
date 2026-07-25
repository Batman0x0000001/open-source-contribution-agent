from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

from osc_agent.runtime.completion import CompletionEvidenceStopHook
from osc_agent.runtime.hooks import StopHookPayload
from osc_agent.runtime.models import (
    CompletionRequirements,
    RuntimeMessage,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
)
from osc_agent.tools.git import git_workspace_fingerprint


def initialize_repository(root: Path) -> str:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    (root / "x.py").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "x.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=root, check=True)
    (root / "x.py").write_text("changed\n", encoding="utf-8")
    return git_workspace_fingerprint(repo_root=root)


def evidence(root: Path, data: dict) -> dict:
    return {**data, "workspace_fingerprint": git_workspace_fingerprint(repo_root=root)}


def snapshot(root: Path) -> dict:
    return evidence(
        root,
        {"base_commit": "abc", "files": [{"path": "x.py"}], "has_changes": True},
    )


def context(root: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="completion",
        working_directory=str(root),
        repository_root=str(root),
        state_directory=str(root / "state"),
        completion_requirements=CompletionRequirements(
            required_evidence=frozenset(
                {"successful_test", "git_change_snapshot"}
            ),
            waivable_evidence=frozenset({"successful_test"}),
        ),
    )


def independent_context(root: Path) -> ToolUseContext:
    return context(root).model_copy(
        update={
            "completion_requirements": CompletionRequirements(
                required_evidence=frozenset(
                    {
                        "successful_test",
                        "independent_verification",
                        "git_change_snapshot",
                    }
                ),
                waivable_evidence=frozenset(
                    {"successful_test", "independent_verification"}
                ),
            )
        }
    )


def completed(call_id: str, name: str, input: dict, data: dict) -> list[RuntimeMessage]:
    return [
        RuntimeMessage(
            role="assistant",
            content=[ToolUseBlock(id=call_id, name=name, input=input)],
        ),
        RuntimeMessage(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id=call_id,
                    content={"data": data, "error": None},
                )
            ],
        ),
    ]


def evaluate(messages: list[RuntimeMessage], root: Path):
    return asyncio.run(
        CompletionEvidenceStopHook()(
            StopHookPayload(messages=messages),
            context(root),
        )
    )


def evaluate_independent(messages: list[RuntimeMessage], root: Path):
    return asyncio.run(
        CompletionEvidenceStopHook()(
            StopHookPayload(messages=messages),
            independent_context(root),
        )
    )


def verification_result(root: Path, child_session_id: str, verdict: str) -> dict:
    return {
        "agent": "any-verification-provider",
        "status": "completed",
        "child_session_id": child_session_id,
        "result": {
            "evidence_type": "independent_verification",
            "verdict": verdict,
            "summary": "report",
            "checks": [],
            "risks": [],
            "unverified": [],
        },
        "error": None,
        "workspace_fingerprint": git_workspace_fingerprint(repo_root=root),
    }


def test_completion_requires_test_and_snapshot_after_latest_edit(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    messages = [
        *completed("edit", "write_file", {}, {"path": "x.py"}),
        *completed(
            "test",
            "powershell",
            {},
            evidence(tmp_path, {"success": True, "command_kind": "test"}),
        ),
        *completed(
            "diff",
            "git_diff",
            {},
            snapshot(tmp_path),
        ),
    ]

    assert evaluate(messages, tmp_path).blocking_reasons == []

    (tmp_path / "x.py").write_text("changed again\n", encoding="utf-8")
    stale = [*messages, *completed("later-edit", "edit_file", {}, {"path": "x.py"})]
    reasons = evaluate(stale, tmp_path).blocking_reasons
    assert any("successful test" in reason for reason in reasons)
    assert any("git_diff" in reason for reason in reasons)


def test_empty_snapshot_never_satisfies_change_evidence(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    messages = [
        *completed(
            "test",
            "powershell",
            {},
            evidence(tmp_path, {"success": True, "command_kind": "test"}),
        ),
        *completed(
            "diff",
            "git_diff",
            {},
            evidence(
                tmp_path,
                {"base_commit": "abc", "files": [], "has_changes": False},
            ),
        ),
    ]

    assert any(
        "non-empty" in reason
        for reason in evaluate(messages, tmp_path).blocking_reasons
    )


def test_verification_waiver_must_be_explicit_and_snapshot_remains_required(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    waiver = {
        "questions": [
            {
                "id": "waiver",
                "purpose": "verification_waiver",
                "reason": "The repository has no executable test suite.",
                "alternative_checks": ["Reviewed rendered documentation"],
                "risks": ["Rendering may differ in CI"],
            }
        ],
        "workspace_fingerprint": git_workspace_fingerprint(repo_root=tmp_path),
        "answers": [
            {
                "question_id": "waiver",
                "selected_option_id": "proceed_without_tests",
                "custom_text": None,
            }
        ],
    }
    without_snapshot = [
        *completed("edit", "write_file", {}, {"path": "docs/x.md"}),
        *completed("waiver", "ask_user_question", {}, waiver),
    ]
    reasons = evaluate(without_snapshot, tmp_path).blocking_reasons
    assert not any("successful test" in reason for reason in reasons)
    assert any("git_diff" in reason for reason in reasons)

    complete = [
        *without_snapshot,
        *completed(
            "diff",
            "git_diff",
            {},
            snapshot(tmp_path),
        ),
    ]
    assert evaluate(complete, tmp_path).blocking_reasons == []


def test_free_text_does_not_approve_verification_waiver(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    messages = [
        *completed("edit", "write_file", {}, {"path": "x.py"}),
        *completed(
            "waiver",
            "ask_user_question",
            {},
            {
                "questions": [
                    {
                        "id": "waiver",
                        "purpose": "verification_waiver",
                        "reason": "Unavailable",
                        "alternative_checks": ["Static review"],
                        "risks": ["Unknown runtime behavior"],
                    }
                ],
                "answers": [
                    {
                        "question_id": "waiver",
                        "selected_option_id": None,
                        "custom_text": "okay",
                    }
                ],
                "workspace_fingerprint": git_workspace_fingerprint(repo_root=tmp_path),
            },
        ),
    ]
    assert any(
        "successful test" in reason
        for reason in evaluate(messages, tmp_path).blocking_reasons
    )


def test_independent_verification_requires_primary_test_then_pass_then_snapshot(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    edit = completed("edit", "write_file", {}, {"path": "x.py"})
    test = completed(
        "test",
        "powershell",
        {},
        evidence(tmp_path, {"success": True, "command_kind": "test"}),
    )
    verify = completed(
        "verify",
        "agent",
        {},
        verification_result(tmp_path, "verify-child", "PASS"),
    )
    final_snapshot = completed(
        "diff",
        "git_diff",
        {},
        snapshot(tmp_path),
    )

    missing = evaluate_independent([*edit, *test, *final_snapshot], tmp_path)
    assert any("independent verification PASS" in reason for reason in missing.blocking_reasons)

    stale_order = evaluate_independent([*edit, *verify, *test, *final_snapshot], tmp_path)
    assert any("independent verification PASS" in reason for reason in stale_order.blocking_reasons)

    missing_final_snapshot = evaluate_independent([*edit, *test, *verify], tmp_path)
    assert any("final git_diff" in reason for reason in missing_final_snapshot.blocking_reasons)

    assert evaluate_independent([*edit, *test, *verify, *final_snapshot], tmp_path).blocking_reasons == []


def test_latest_fail_or_later_edit_invalidates_independent_pass(tmp_path: Path) -> None:
    initialize_repository(tmp_path)
    messages = [
        *completed("edit", "edit_file", {}, {"path": "x.py"}),
        *completed(
            "test",
            "powershell",
            {},
            evidence(tmp_path, {"success": True, "command_kind": "test"}),
        ),
        *completed("pass", "agent", {}, verification_result(tmp_path, "pass-child", "PASS")),
        *completed("fail", "agent", {}, verification_result(tmp_path, "fail-child", "FAIL")),
        *completed(
            "diff",
            "git_diff",
            {},
            snapshot(tmp_path),
        ),
    ]
    reasons = evaluate_independent(messages, tmp_path).blocking_reasons
    assert any("independent verification PASS" in reason for reason in reasons)

    (tmp_path / "x.py").write_text("later\n", encoding="utf-8")
    later_edit = [*messages[:6], *completed("later-edit", "write_file", {}, {"path": "y.py"})]
    reasons = evaluate_independent(later_edit, tmp_path).blocking_reasons
    assert any("successful test" in reason for reason in reasons)
    assert any("independent verification PASS" in reason for reason in reasons)


def test_partial_waiver_must_bind_latest_child_and_precede_snapshot(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    prefix = [
        *completed("edit", "write_file", {}, {"path": "x.py"}),
        *completed(
            "test",
            "powershell",
            {},
            evidence(tmp_path, {"success": True, "command_kind": "test"}),
        ),
        *completed(
            "partial",
            "agent",
            {},
            verification_result(tmp_path, "partial-child", "PARTIAL"),
        ),
    ]

    def waiver(child: str, selected: str | None = "proceed_with_partial_verification"):
        return completed(
            "waiver",
            "ask_user_question",
            {},
            {
                "questions": [
                    {
                        "id": "partial-waiver",
                        "purpose": "independent_verification_waiver",
                        "related_child_session_id": child,
                        "reason": "Docker is unavailable.",
                        "alternative_checks": ["Unit tests passed"],
                        "unverified_checks": ["Container startup"],
                        "risks": ["Container configuration may fail"],
                    }
                ],
                "answers": [
                    {
                        "question_id": "partial-waiver",
                        "selected_option_id": selected,
                        "custom_text": "continue" if selected is None else None,
                    }
                ],
                "workspace_fingerprint": git_workspace_fingerprint(repo_root=tmp_path),
            },
        )

    final_snapshot = completed(
        "diff",
        "git_diff",
        {},
        snapshot(tmp_path),
    )
    wrong = evaluate_independent(
        [*prefix, *waiver("wrong-child"), *final_snapshot],
        tmp_path,
    )
    assert any("independent verification PASS" in reason for reason in wrong.blocking_reasons)

    free_text = evaluate_independent(
        [*prefix, *waiver("partial-child", None), *final_snapshot],
        tmp_path,
    )
    assert any("independent verification PASS" in reason for reason in free_text.blocking_reasons)

    without_snapshot = evaluate_independent(
        [*prefix, *waiver("partial-child")],
        tmp_path,
    )
    assert any("final git_diff" in reason for reason in without_snapshot.blocking_reasons)

    complete = evaluate_independent(
        [*prefix, *waiver("partial-child"), *final_snapshot],
        tmp_path,
    )
    assert complete.blocking_reasons == []

    no_report = evaluate_independent(
        [*prefix[:-2], *waiver("partial-child"), *final_snapshot],
        tmp_path,
    )
    assert any("independent verification PASS" in reason for reason in no_report.blocking_reasons)
