from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from osc_agent.agents.definitions import AgentInvocation, AgentRunResult
from osc_agent.agents.registry import AgentRegistry
from osc_agent.agents.tool import AgentTool, AgentToolInput
from osc_agent.agents.verify import (
    VERIFY_TOOLS,
    VerificationReport,
    build_verify_registration,
)
from osc_agent.runtime.models import CapabilityScope, ToolUseContext


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=root, check=True)


def context(root: Path) -> ToolUseContext:
    return ToolUseContext(
        session_id="parent",
        working_directory=str(root),
        repository_root=str(root),
        state_directory=str(root.parent / "state"),
        capabilities=CapabilityScope(allowed_tools=frozenset({"agent", *VERIFY_TOOLS})),
    )


def report(*, verdict: str, checks: list[dict], risks=None, unverified=None) -> dict:
    return {
        "evidence_type": "independent_verification",
        "verdict": verdict,
        "summary": "independent result",
        "checks": checks,
        "risks": risks or [],
        "unverified": unverified or [],
    }


def test_verify_registration_is_minimal_read_only_and_bounded() -> None:
    registration = build_verify_registration(model="test-model")

    assert registration.definition.name == "verify"
    assert registration.definition.context_policy == "minimal"
    assert registration.definition.capabilities.allowed_tools == VERIFY_TOOLS
    assert registration.definition.config.max_rounds == 12
    assert registration.definition.config.max_total_tokens == 60_000
    assert registration.definition.config.deadline_seconds == 900
    assert registration.read_only is True
    assert registration.concurrency_safe is False
    assert registration.max_parallel == 1
    assert "agent" not in VERIFY_TOOLS
    assert "write_file" not in VERIFY_TOOLS


def test_verification_report_enforces_verdict_semantics() -> None:
    valid = VerificationReport.model_validate(
        report(
            verdict="PASS",
            checks=[
                {
                    "name": "boundary",
                    "command": "python -m pytest",
                    "exit_code": 0,
                    "output_excerpt": "passed",
                    "result": "pass",
                    "adversarial": True,
                }
            ],
        )
    )
    assert valid.verdict == "PASS"

    with pytest.raises(ValidationError, match="adversarial"):
        VerificationReport.model_validate(
            report(
                verdict="PASS",
                checks=[
                    {
                        "name": "suite",
                        "command": "python -m pytest",
                        "exit_code": 0,
                        "output_excerpt": "passed",
                        "result": "pass",
                    }
                ],
            )
        )
    with pytest.raises(ValidationError, match="must use verdict FAIL"):
        VerificationReport.model_validate(
            report(
                verdict="PARTIAL",
                checks=[
                    {
                        "name": "suite",
                        "command": "python -m pytest",
                        "exit_code": 1,
                        "output_excerpt": "failed",
                        "result": "fail",
                    }
                ],
                risks=["regression"],
                unverified=["integration"],
            )
        )
    with pytest.raises(ValidationError, match="explicit risks"):
        VerificationReport.model_validate(
            report(
                verdict="PARTIAL",
                checks=[
                    {
                        "name": "integration",
                        "command": "docker compose up",
                        "exit_code": None,
                        "output_excerpt": "docker unavailable",
                        "result": "blocked",
                    }
                ],
                unverified=["integration"],
            )
        )


def test_verify_agent_returns_typed_report_without_parent_transcript(tmp_path: Path) -> None:
    initialize_repository(tmp_path)

    class Runner:
        invocation: AgentInvocation | None = None

        async def run(self, invocation: AgentInvocation) -> AgentRunResult:
            self.invocation = invocation
            return AgentRunResult(
                session_id="verify-child",
                status="completed",
                output=__import__("json").dumps(
                    report(
                        verdict="PASS",
                        checks=[
                            {
                                "name": "boundary",
                                "command": "python -m pytest",
                                "exit_code": 0,
                                "output_excerpt": "passed",
                                "result": "pass",
                                "adversarial": True,
                            }
                        ],
                    )
                ),
            )

    runner = Runner()
    registry = AgentRegistry([build_verify_registration(model="test-model")])
    result = asyncio.run(
        AgentTool(runner, registry).call(
            AgentToolInput(
                agent="verify",
                task="Independently verify the fix",
                arguments={
                    "original_goal": "Fix boundary handling",
                    "implementation_summary": "Validated the boundary before indexing",
                    "focus_areas": ["empty input"],
                },
            ),
            context(tmp_path),
        )
    )

    assert result.error is None
    assert result.data["result"]["verdict"] == "PASS"
    assert runner.invocation is not None
    assert runner.invocation.parent_messages == []
    assert runner.invocation.working_directory == str(tmp_path)


def test_read_only_agent_guard_fails_closed_and_preserves_changes(tmp_path: Path) -> None:
    initialize_repository(tmp_path)

    class MutatingRunner:
        async def run(self, invocation: AgentInvocation) -> AgentRunResult:
            (Path(invocation.working_directory) / "unexpected.bin").write_bytes(b"\x00\x01")
            return AgentRunResult(session_id="child", status="completed", output="{}")

    registry = AgentRegistry([build_verify_registration(model="test-model")])
    result = asyncio.run(
        AgentTool(MutatingRunner(), registry).call(
            AgentToolInput(
                agent="verify",
                task="verify",
                arguments={
                    "original_goal": "goal",
                    "implementation_summary": "summary",
                },
            ),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "AGENT_READ_ONLY_VIOLATION"
    assert (tmp_path / "unexpected.bin").is_file()

    outside_git = tmp_path / "outside"
    outside_git.mkdir()
    failed = asyncio.run(
        AgentTool(MutatingRunner(), registry).call(
            AgentToolInput(
                agent="verify",
                task="verify",
                arguments={
                    "original_goal": "goal",
                    "implementation_summary": "summary",
                },
            ),
            context(outside_git),
        )
    )
    assert failed.error and failed.error.code == "AGENT_READ_ONLY_GUARD_FAILED"
