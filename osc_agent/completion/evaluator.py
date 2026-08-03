"""以中立输入评估 Agent transcript 的完成证据与配置验证。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import Field

from osc_agent.completion.models import CompletionRequirements
from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.messages import RuntimeMessage
from osc_agent.runtime.messages import ToolResultBlock, ToolUseBlock
from osc_agent.workspaces.git_state import git_workspace_fingerprint


class CompletionEvaluation(FrozenContractModel):
    messages: tuple[RuntimeMessage, ...]
    workspace_root: str = Field(min_length=1)
    requirements: CompletionRequirements = Field(default_factory=CompletionRequirements)
    validation_commands: tuple[str, ...] = ()


class CompletionReport(FrozenContractModel):
    blocking_reasons: tuple[str, ...] = ()


class CompletionEvaluator:
    """从权威 transcript 推导完成结果，不依赖 Runtime Hook。"""

    async def evaluate(self, evaluation: CompletionEvaluation) -> CompletionReport:
        required = evaluation.requirements.required_evidence
        if not required:
            return CompletionReport()
        try:
            current_fingerprint = await asyncio.to_thread(
                git_workspace_fingerprint,
                repo_root=Path(evaluation.workspace_root),
            )
        except (OSError, ValueError) as exc:
            return CompletionReport(
                blocking_reasons=(
                    f"Unable to verify the current Git workspace fingerprint: {exc}"
                ,)
            )

        calls: dict[str, tuple[int, ToolUseBlock]] = {}
        completed: list[tuple[int, ToolUseBlock, dict]] = []
        for index, message in enumerate(evaluation.messages):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    calls[block.id] = (index, block)
                elif isinstance(block, ToolResultBlock):
                    call = calls.get(block.tool_use_id)
                    if call is not None and isinstance(block.content, dict):
                        completed.append((index, call[1], block.content))

        successful_test = max(
            (
                (index, str(content["data"].get("workspace_fingerprint") or ""))
                for index, call, content in completed
                if call.name == "bash"
                and isinstance(content.get("data"), dict)
                and content["data"].get("success") is True
                and content["data"].get("command_kind") == "test"
            ),
            default=None,
        )
        waiver = max(
            (
                (
                    index,
                    str(content["data"].get("workspace_fingerprint") or ""),
                )
                for index, call, content in completed
                if call.name == "ask_user_question"
                and _has_verification_waiver(content)
            ),
            default=None,
        )
        primary_candidates = [
            item
            for item in (
                successful_test,
                waiver
                if "successful_test"
                in evaluation.requirements.waivable_evidence
                else None,
            )
            if item is not None
        ]
        verification = max(primary_candidates, default=None)
        verification_index = verification[0] if verification is not None else -1
        verification_fingerprint = verification[1] if verification is not None else ""
        independent_reports = [
            (
                index,
                str(content["data"].get("child_session_id") or ""),
                str(content["data"]["result"].get("verdict") or ""),
                str(content["data"].get("workspace_fingerprint") or ""),
            )
            for index, _call, content in completed
            if _independent_verification_report(content)
        ]
        latest_independent = max(independent_reports, default=None)
        independent_verification = (
            (latest_independent[0], latest_independent[3])
            if latest_independent is not None
            and latest_independent[2] == "PASS"
            else None
        )
        independent_waiver = max(
            (
                (
                    index,
                    str(content["data"].get("workspace_fingerprint") or ""),
                )
                for index, _call, content in completed
                if latest_independent is not None
                and latest_independent[2] == "PARTIAL"
                and index > latest_independent[0]
                and _has_independent_verification_waiver(
                    content,
                    child_session_id=latest_independent[1],
                )
            ),
            default=None,
        )
        independent_candidates = [
            item
            for item in (
                independent_verification,
                independent_waiver
                if "independent_verification"
                in evaluation.requirements.waivable_evidence
                else None,
            )
            if item is not None
        ]
        independent_evidence = max(independent_candidates, default=None)
        independent_index = (
            independent_evidence[0] if independent_evidence is not None else -1
        )
        independent_fingerprint = (
            independent_evidence[1] if independent_evidence is not None else ""
        )
        snapshot = max(
            (
                (
                    index,
                    str(content["data"].get("workspace_fingerprint") or ""),
                    bool(content["data"].get("has_changes")),
                )
                for index, call, content in completed
                if call.name == "git_diff"
                and not content.get("error")
                and isinstance(content.get("data"), dict)
                and bool(content["data"].get("base_commit"))
            ),
            default=None,
        )
        issue_plan = max(
            (
                index
                for index, call, content in completed
                if call.name == "submit_issue_plan"
                and not content.get("error")
                and isinstance(content.get("data"), dict)
                and content["data"].get("status") in {"ready", "blocked"}
            ),
            default=None,
        )
        delivery_draft = max(
            (
                (
                    index,
                    str(content["data"].get("workspace_fingerprint") or ""),
                )
                for index, call, content in completed
                if call.name == "submit_delivery_draft"
                and not content.get("error")
                and isinstance(content.get("data"), dict)
            ),
            default=None,
        )

        reasons: list[str] = []
        if (
            "successful_test" in required
            and (
                verification is None
                or verification_fingerprint != current_fingerprint
            )
        ):
            reasons.append(
                "No successful test or explicit test waiver is bound to the current Git workspace."
            )
        if (
            "independent_verification" in required
            and (
                independent_evidence is None
                or independent_index <= verification_index
                or independent_fingerprint != current_fingerprint
            )
        ):
            if (
                independent_evidence is not None
                and independent_index <= verification_index
            ):
                reasons.append(
                    "The latest primary test invalidated the earlier independent verification "
                    "PASS. Run Verify again now, then do not rerun tests unless the workspace "
                    "changes."
                )
            else:
                reasons.append(
                    "No independent verification PASS or explicit PARTIAL waiver is bound to the "
                    "current Git workspace after the primary test. Run Verify now."
                )
        if (
            "git_change_snapshot" in required
            and (
                snapshot is None
                or snapshot[0] <= max(verification_index, independent_index)
                or snapshot[1] != current_fingerprint
                or not snapshot[2]
            )
        ):
            reasons.append(
                "No non-empty final git_diff snapshot is bound to the current Git workspace after "
                "the primary test and independent verification evidence. Run git_diff now, then "
                "do not test, Verify, or mutate the workspace."
            )
        if "issue_plan" in required and issue_plan is None:
            reasons.append("No strict IssuePlanArtifact has been submitted for this planning Session.")
        if (
            "delivery_draft" in required
            and (
                delivery_draft is None
                or snapshot is None
                or delivery_draft[0] <= snapshot[0]
                or delivery_draft[1] != current_fingerprint
            )
        ):
            reasons.append(
                "No DeliveryDraft bound to the current workspace was submitted after the final "
                "git snapshot. Call submit_delivery_draft with draft content only; the trusted "
                "worker binds Job and workspace identity, then stop."
            )
        reasons.extend(
            await _configured_validation_reasons(
                evaluation,
                current_fingerprint=current_fingerprint,
            )
        )
        return CompletionReport(blocking_reasons=tuple(reasons))


async def _configured_validation_reasons(
    evaluation: CompletionEvaluation,
    *,
    current_fingerprint: str,
) -> list[str]:
    if (
        not evaluation.validation_commands
        or "successful_test" not in evaluation.requirements.required_evidence
    ):
        return []
    calls: dict[str, tuple[int, ToolUseBlock]] = {}
    success: dict[str, tuple[int, str]] = {}
    last_write = -1
    for index, message in enumerate(evaluation.messages):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                calls[block.id] = (index, block)
            elif isinstance(block, ToolResultBlock):
                call = calls.get(block.tool_use_id)
                if call is None or not isinstance(block.content, dict):
                    continue
                if call[1].name in {"write_file", "edit_file"} and not block.content.get(
                    "error"
                ):
                    last_write = index
                data = block.content.get("data")
                if (
                    call[1].name == "bash"
                    and isinstance(data, dict)
                    and data.get("success") is True
                ):
                    command = data.get("command")
                    fingerprint = data.get("workspace_fingerprint")
                    if isinstance(command, str) and isinstance(fingerprint, str):
                        success[command.strip()] = (index, fingerprint)
    missing = [
        command
        for command in evaluation.validation_commands
        if command.strip() not in success
        or success[command.strip()][0] <= last_write
        or success[command.strip()][1] != current_fingerprint
    ]
    if not missing:
        return []
    return [
        "Configured validation commands have not all succeeded on the current workspace: "
        + ", ".join(missing)
    ]


def _has_verification_waiver(content: dict) -> bool:
    data = content.get("data")
    if not isinstance(data, dict):
        return False
    questions = data.get("questions")
    answers = data.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, list):
        return False
    waiver_ids = {
        str(question.get("id"))
        for question in questions
        if isinstance(question, dict)
        and question.get("purpose") == "verification_waiver"
        and question.get("reason")
        and question.get("alternative_checks")
        and question.get("risks")
    }
    return any(
        isinstance(answer, dict)
        and str(answer.get("question_id")) in waiver_ids
        and answer.get("selected_option_id") == "proceed_without_tests"
        for answer in answers
    )


def _independent_verification_report(content: dict) -> bool:
    data = content.get("data")
    if not isinstance(data, dict) or data.get("status") != "completed":
        return False
    result = data.get("result")
    return (
        isinstance(result, dict)
        and result.get("evidence_type") == "independent_verification"
        and result.get("verdict") in {"PASS", "FAIL", "PARTIAL"}
        and bool(data.get("child_session_id"))
    )


def _has_independent_verification_waiver(
    content: dict,
    *,
    child_session_id: str,
) -> bool:
    data = content.get("data")
    if not isinstance(data, dict):
        return False
    questions = data.get("questions")
    answers = data.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, list):
        return False
    waiver_ids = {
        str(question.get("id"))
        for question in questions
        if isinstance(question, dict)
        and question.get("purpose") == "independent_verification_waiver"
        and question.get("related_child_session_id") == child_session_id
        and question.get("reason")
        and question.get("alternative_checks")
        and question.get("unverified_checks")
        and question.get("risks")
    }
    return any(
        isinstance(answer, dict)
        and str(answer.get("question_id")) in waiver_ids
        and answer.get("selected_option_id") == "proceed_with_partial_verification"
        for answer in answers
    )
