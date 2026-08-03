"""验证Bot 权限与完成策略的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import asyncio
from pathlib import Path
import subprocess

from osc_agent.completion.hooks import CompletionStopHook
from osc_agent.runtime.hooks import StopHookPayload
from osc_agent.completion.models import CompletionRequirements


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=root, check=True)


def context(root: Path, *, requires_successful_test: bool) -> tool_context:
    required = frozenset({"successful_test"}) if requires_successful_test else frozenset()
    return tool_context(
        session_id="validation-hook",
        working_directory=str(root),
        state_directory=str(root / "state"),
        completion_requirements=CompletionRequirements(required_evidence=required),
    )


def test_configured_validation_hook_ignores_child_sessions_without_test_evidence(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    hook = CompletionStopHook(
        validation_commands=("python -m pytest", "python -m pip check")
    )

    result = asyncio.run(
        hook(StopHookPayload(messages=[]), context(tmp_path, requires_successful_test=False))
    )

    assert result.blocking_reasons == []


def test_configured_validation_hook_still_blocks_implementation_session(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    hook = CompletionStopHook(validation_commands=("python -m pytest",))

    result = asyncio.run(
        hook(StopHookPayload(messages=[]), context(tmp_path, requires_successful_test=True))
    )

    assert result.blocking_reasons == [
        "No successful test or explicit test waiver is bound to the current Git workspace.",
        "Configured validation commands have not all succeeded on the current workspace: "
        "python -m pytest",
    ]
