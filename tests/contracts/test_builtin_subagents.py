"""验证内置 Explore 与 Verify 子 Agent 的契约和边界。"""

from __future__ import annotations

from osc_agent.runtime.state import CapabilityScope

from tests.runtime_factories import tool_context

import asyncio
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from osc_agent.subagents.builtins.verify import (
    VERIFY_TOOLS,
    VerificationReport,
    build_verify_subagent,
)
from osc_agent.subagents.models import SubagentRequest, SubagentRunResult
from osc_agent.subagents.registry import SubagentRegistry
from osc_agent.subagents.tool import AgentTool, AgentToolInput

from osc_agent.configuration.runtime import default_runtime_config_path, load_runtime_config


VERIFY_CONFIG = load_runtime_config(default_runtime_config_path()).agents.verify.to_query_config()


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=root, check=True)


def context(root: Path) -> tool_context:
    return tool_context(
        session_id="parent",
        working_directory=str(root),
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
    registration = build_verify_subagent(model="test-model", config=VERIFY_CONFIG)

    assert registration.definition.name == "verify"
    assert registration.definition.context_policy == "minimal"
    assert registration.definition.capabilities.allowed_tools == VERIFY_TOOLS
    assert registration.definition.config.max_rounds == 16
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
        name: str | None = None
        request: SubagentRequest | None = None

        async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
            self.name = name
            self.request = request
            return SubagentRunResult(
                session_id="verify-child",
                status="completed",
                output=(
                    "Verification complete.\n```json\n"
                    + __import__("json").dumps(
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
                    + "\n```"
                ),
            )

    runner = Runner()
    registry = SubagentRegistry(
        [build_verify_subagent(model="test-model", config=VERIFY_CONFIG)]
    )
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
    assert runner.name == "verify"
    assert runner.request is not None
    assert runner.request.working_directory == str(tmp_path)


def test_verify_prompt_states_semantic_output_constraints() -> None:
    registration = build_verify_subagent(model="test-model", config=VERIFY_CONFIG)
    prompt = registration.prompt_builder(
        registration.input_model.model_validate(
            {
                "original_goal": "goal",
                "implementation_summary": "summary",
            }
        )
    )

    assert "Do not wrap the JSON in Markdown" in prompt
    assert "PASS requires an empty unverified list" in prompt
    assert "blocked checks require exit_code null" in prompt


def test_read_only_agent_guard_fails_closed_and_preserves_changes(tmp_path: Path) -> None:
    initialize_repository(tmp_path)

    class MutatingRunner:
        async def run(self, name: str, request: SubagentRequest) -> SubagentRunResult:
            (Path(request.working_directory) / "unexpected.bin").write_bytes(b"\x00\x01")
            return SubagentRunResult(session_id="child", status="completed", output="{}")

    registry = SubagentRegistry(
        [build_verify_subagent(model="test-model", config=VERIFY_CONFIG)]
    )
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
