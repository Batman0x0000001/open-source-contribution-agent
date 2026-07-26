from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess

from osc_agent.runtime.models import ApprovalResponse, Ask, ToolUseBlock, ToolUseContext
from osc_agent.runtime.tool_execution import ToolExecutionDependencies, ToolExecutor
from tests.contracts.registry_factory import build_test_tool_registry as build_core_tool_registry
import osc_agent.tools.shell as shell_module
import osc_agent.tools.process as process_module
from osc_agent.tools.shell import ShellInput, ShellTool
from osc_agent.tools.process import (
    CommandKind,
    build_subprocess_environment,
    classify_command,
)


def context(root: Path) -> ToolUseContext:
    return ToolUseContext(session_id="session-1", working_directory=str(root), repository_root=str(root), state_directory=str(root / "state"))


def test_shell_classifies_behavior_from_each_input() -> None:
    tool = ShellTool()

    assert tool.name == "powershell"
    assert Path(tool.executable).stem.casefold() == "pwsh"
    assert tool.is_read_only(ShellInput(command="git status")) is True
    assert tool.is_concurrency_safe(ShellInput(command="rg pattern src")) is True
    assert tool.is_read_only(ShellInput(command="git status && git push")) is False
    assert tool.is_destructive(ShellInput(command="python build.py")) is True


def test_shell_read_only_classifier_rejects_execution_writes_and_path_escape() -> None:
    tool = ShellTool()

    assert tool.is_read_only(ShellInput(command="rg --pre malicious pattern")) is False
    assert tool.is_read_only(ShellInput(command="rg --follow pattern .")) is False
    assert tool.is_read_only(ShellInput(command="rg -L pattern .")) is False
    assert tool.is_read_only(ShellInput(command="git diff --output=result.patch")) is False
    assert tool.is_read_only(ShellInput(command="Get-Content .\\README.md")) is False
    assert tool.is_read_only(ShellInput(command="Get-Content ..\\secret.txt")) is False
    assert tool.is_read_only(ShellInput(command="Get-Content C:\\secret.txt")) is False


def test_read_only_shell_executes_without_approval(tmp_path: Path) -> None:
    executor = ToolExecutor(build_core_tool_registry())

    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="shell-1", name="powershell", input={"command": "git status --short"}),
            context(tmp_path),
        )
    )

    assert result.error is None
    assert result.data["command"] == "git status --short"
    assert isinstance(result.data["exit_code"], int)


def test_non_read_only_shell_requires_one_explicit_approval(tmp_path: Path) -> None:
    approvals = 0
    request: Ask | None = None

    async def approve(decision: Ask) -> ApprovalResponse:
        nonlocal approvals, request
        approvals += 1
        request = decision
        return ApprovalResponse(choice="allow_once")

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="shell-1", name="powershell", input={"command": "python -c \"print('ok')\""}),
            context(tmp_path),
        )
    )

    assert approvals == 1
    assert request is not None
    assert request.tool_name == "powershell"
    assert request.working_directory == str(tmp_path)
    assert request.risk == "process"
    assert request.preview["command"] == "python -c \"print('ok')\""
    assert result.error is None
    assert "ok" in result.data["stdout"]
    assert result.data["command_kind"] == "other"


def test_denied_shell_pattern_is_rejected_before_permission_prompt(tmp_path: Path) -> None:
    approvals = 0

    async def approve(decision: Ask) -> ApprovalResponse:
        nonlocal approvals
        approvals += 1
        return ApprovalResponse(choice="allow_once")

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    result = asyncio.run(
        executor.execute(
            ToolUseBlock(id="shell-1", name="powershell", input={"command": "git push origin main"}),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert approvals == 0


def test_powershell_provider_paths_are_hard_rejected_before_approval(tmp_path: Path) -> None:
    approvals = 0

    async def approve(_decision: Ask) -> ApprovalResponse:
        nonlocal approvals
        approvals += 1
        return ApprovalResponse(choice="allow_once")

    executor = ToolExecutor(
        build_core_tool_registry(),
        dependencies=ToolExecutionDependencies(approval_handler=approve),
    )
    commands = [
        "Get-Content Env:ANTHROPIC_API_KEY",
        "Write-Output $env:PATH",
        "Write-Output ${Env:PATH}",
        "Get-Variable Variable:secret",
        "Get-Item Registry::HKEY_CURRENT_USER\\Software",
        "Get-ChildItem Cert:\\CurrentUser",
        "Get-Item WSMan:\\localhost",
    ]

    for index, command in enumerate(commands):
        result = asyncio.run(
            executor.execute(
                ToolUseBlock(
                    id=f"provider-{index}",
                    name="powershell",
                    input={"command": command},
                ),
                context(tmp_path),
            )
        )
        assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert approvals == 0


def test_powershell_ast_rejects_git_alias_and_hidden_command_runners(
    tmp_path: Path,
) -> None:
    executor = ToolExecutor(build_core_tool_registry())
    commands = [
        "git -c alias.publish=push publish origin HEAD",
        "$command = 'git'; & $command push origin HEAD",
        "cmd /c git push origin HEAD",
        "pwsh -Command 'git push origin HEAD'",
        "gh pr create --fill",
    ]

    for index, command in enumerate(commands):
        result = asyncio.run(
            executor.execute(
                ToolUseBlock(
                    id=f"blocked-{index}",
                    name="powershell",
                    input={"command": command},
                ),
                context(tmp_path),
            )
        )
        assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"


def test_powershell_ast_allows_direct_read_only_git_command(tmp_path: Path) -> None:
    result = asyncio.run(
        ToolExecutor(build_core_tool_registry()).execute(
            ToolUseBlock(
                id="read-only-git",
                name="powershell",
                input={"command": "git status --short"},
            ),
            context(tmp_path),
        )
    )

    assert result.error is None


def test_powershell_process_does_not_use_shell_true() -> None:
    source = (Path(__file__).resolve().parents[2] / "osc_agent" / "tools" / "process.py").read_text(encoding="utf-8")
    assert "create_subprocess_exec" in source
    assert "shell=True" not in source


def test_subprocess_environment_is_allowlist_only_and_credentials_are_never_forwarded() -> None:
    source = {
        "PATH": os.environ.get("PATH", ""),
        "SystemRoot": os.environ.get("SystemRoot", "C:\\Windows"),
        "CUSTOM_BUILD_FLAG": "enabled",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "GITHUB_TOKEN": "github-secret",
        "CUSTOM_PASSWORD": "password-secret",
        "RIPGREP_CONFIG_PATH": "unsafe-rg-config",
        "UNLISTED_VALUE": "hidden",
    }

    environment = build_subprocess_environment(
        [
            "CUSTOM_BUILD_FLAG",
            "ANTHROPIC_API_KEY",
            "GITHUB_TOKEN",
            "CUSTOM_PASSWORD",
            "RIPGREP_CONFIG_PATH",
        ],
        source=source,
    )

    assert environment["CUSTOM_BUILD_FLAG"] == "enabled"
    assert environment["PATH"] == source["PATH"]
    assert "ANTHROPIC_API_KEY" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "CUSTOM_PASSWORD" not in environment
    assert "RIPGREP_CONFIG_PATH" not in environment
    assert "UNLISTED_VALUE" not in environment


def test_powershell_ast_analysis_uses_filtered_environment(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")

    def fake_run(*args, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"errors":[],"commands":[{"name":"git","text":"git status"}]}',
            stderr="",
        )

    monkeypatch.setattr(shell_module.subprocess, "run", fake_run)

    analysis = shell_module._analyze_powershell("git status", executable="pwsh")

    assert analysis["errors"] == []
    assert captured["OSC_AGENT_COMMAND"] == "git status"
    assert "ANTHROPIC_API_KEY" not in captured
    assert "GITHUB_TOKEN" not in captured


def test_command_classification_covers_common_project_verification() -> None:
    assert classify_command("python -m pytest -q") is CommandKind.TEST
    assert classify_command("npm test") is CommandKind.TEST
    assert classify_command("cargo test") is CommandKind.TEST
    assert classify_command("go test ./...") is CommandKind.TEST
    assert classify_command("dotnet test") is CommandKind.TEST
    assert classify_command("mvn test") is CommandKind.TEST
    assert classify_command("ruff check .") is CommandKind.LINT
    assert classify_command("mypy src") is CommandKind.TYPECHECK
    assert classify_command("python script.py") is CommandKind.OTHER


def test_nonzero_powershell_result_is_structured(tmp_path: Path) -> None:
    async def approve(_decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    result = asyncio.run(
        ToolExecutor(
            build_core_tool_registry(),
            dependencies=ToolExecutionDependencies(approval_handler=approve),
        ).execute(
            ToolUseBlock(
                id="failed",
                name="powershell",
                input={"command": "Write-Output bad; exit 3"},
            ),
            context(tmp_path),
        )
    )

    assert result.error is None
    assert result.data["success"] is False
    assert result.data["exit_code"] == 3
    assert "bad" in result.data["stdout"]
    assert result.data["termination_reason"] == "nonzero_exit"


def test_powershell_timeout_is_a_typed_error(tmp_path: Path) -> None:
    async def approve(_decision: Ask) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    result = asyncio.run(
        ToolExecutor(
            build_core_tool_registry(),
            dependencies=ToolExecutionDependencies(approval_handler=approve),
        ).execute(
            ToolUseBlock(
                id="timeout",
                name="powershell",
                input={
                    "command": "Start-Sleep -Seconds 2",
                    "timeout_seconds": 1,
                },
            ),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "POWERSHELL_TIMEOUT"
    assert result.error.retryable is True


def test_cancelling_powershell_propagates_after_process_cleanup(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        tool = ShellTool()
        task = asyncio.create_task(
            tool.call(
                ShellInput(command="Start-Sleep -Seconds 10"),
                context(tmp_path),
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2)
        except asyncio.CancelledError:
            return
        raise AssertionError("PowerShell cancellation was not propagated")

    asyncio.run(run())


def test_windows_process_tree_cleanup_invokes_taskkill_with_tree_and_force(
    monkeypatch,
) -> None:
    if os.name != "nt":
        return
    calls: list[tuple[object, ...]] = []

    class FakeProcess:
        pid = 4321
        returncode = None

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode or 0

    class FakeKiller:
        returncode = 0

        async def communicate(self):
            return b"SUCCESS", b""

    async def fake_create_subprocess_exec(*arguments, **_kwargs):
        calls.append(arguments)
        return FakeKiller()

    monkeypatch.setattr(
        process_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    error = asyncio.run(
        process_module._terminate_process_tree(
            FakeProcess(),  # type: ignore[arg-type]
            {"SystemRoot": "C:\\Windows", "PATH": ""},
        )
    )

    assert error is None
    assert calls == [
        ("C:\\Windows\\System32\\taskkill.exe", "/PID", "4321", "/T", "/F")
    ]
