"""验证共享进程契约、环境策略和宿主 runner。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

from osc_agent.processes.contracts import CommandKind, ProcessRequest
from osc_agent.processes.policy import build_subprocess_environment, classify_command
from osc_agent.processes.runner import DisabledProcessRunner, HostProcessRunner


def _request(root: Path, command: str, *, timeout: float = 5) -> ProcessRequest:
    return ProcessRequest(
        invocation_id="test-process",
        executable=sys.executable,
        command=command,
        repo_root=str(root),
        timeout_seconds=timeout,
        environment=build_subprocess_environment(),
    )


def test_process_policy_filters_secrets_and_classifies_commands() -> None:
    environment = build_subprocess_environment(
        ["CUSTOM_FLAG", "CUSTOM_API_KEY"],
        source={
            "PATH": "bin",
            "CUSTOM_FLAG": "enabled",
            "CUSTOM_API_KEY": "secret",
            "GITHUB_TOKEN": "secret",
        },
    )

    assert environment == {"PATH": "bin", "CUSTOM_FLAG": "enabled"}
    assert classify_command("python -m pytest") == CommandKind.TEST
    assert classify_command("npm run build") == CommandKind.BUILD
    assert classify_command("python script.py") == CommandKind.OTHER


def test_disabled_and_host_runners_share_one_contract(tmp_path: Path) -> None:
    request = _request(tmp_path, "print('ok')")
    disabled = asyncio.run(DisabledProcessRunner().run(request))
    completed = asyncio.run(HostProcessRunner().run(request))

    assert disabled.termination_reason == "disabled"
    assert completed.exit_code == 0
    assert completed.stdout.strip() == "ok"


def test_host_runner_reports_timeout(tmp_path: Path) -> None:
    result = asyncio.run(
        HostProcessRunner().run(
            _request(tmp_path, "import time; time.sleep(5)", timeout=0.05),
        )
    )

    assert result.exit_code == -1
    assert result.termination_reason == "timeout"


def test_host_runner_propagates_cancellation_after_cleanup(tmp_path: Path) -> None:
    async def cancel() -> None:
        task = asyncio.create_task(
            HostProcessRunner().run(
                _request(tmp_path, "import time; time.sleep(5)"),
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
