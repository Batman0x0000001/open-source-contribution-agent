from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

from osc_agent.bot.policy import ConfiguredValidationStopHook
from osc_agent.runtime.hooks import StopHookPayload
from osc_agent.runtime.models import CompletionRequirements, ToolUseContext


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=root, check=True)


def context(root: Path, *, requires_successful_test: bool) -> ToolUseContext:
    required = frozenset({"successful_test"}) if requires_successful_test else frozenset()
    return ToolUseContext(
        session_id="validation-hook",
        working_directory=str(root),
        repository_root=str(root),
        state_directory=str(root / "state"),
        completion_requirements=CompletionRequirements(required_evidence=required),
    )


def test_configured_validation_hook_ignores_child_sessions_without_test_evidence(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    hook = ConfiguredValidationStopHook(("python -m pytest", "python -m pip check"))

    result = asyncio.run(
        hook(StopHookPayload(messages=[]), context(tmp_path, requires_successful_test=False))
    )

    assert result.blocking_reasons == []


def test_configured_validation_hook_still_blocks_implementation_session(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)
    hook = ConfiguredValidationStopHook(("python -m pytest",))

    result = asyncio.run(
        hook(StopHookPayload(messages=[]), context(tmp_path, requires_successful_test=True))
    )

    assert result.blocking_reasons == [
        "Configured validation commands have not all succeeded on the current workspace: "
        "python -m pytest"
    ]
