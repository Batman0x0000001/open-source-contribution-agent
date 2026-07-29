"""验证 Agent 完成前是否执行了配置要求的测试。"""

from __future__ import annotations

from pathlib import Path
import asyncio

from osc_agent.runtime.hooks import StopHookPayload, StopHookResult
from osc_agent.runtime.messages import ToolResultBlock, ToolUseBlock
from osc_agent.runtime.tool_models import ToolUseContext
from osc_agent.workspaces.git_state import git_workspace_fingerprint


class ConfiguredValidationStopHook:
    def __init__(self, commands: tuple[str, ...]) -> None:
        self.commands = commands

    async def __call__(self, payload: StopHookPayload, context: ToolUseContext) -> StopHookResult:
        if "successful_test" not in context.completion_requirements.required_evidence:
            return StopHookResult()
        current = await asyncio.to_thread(
            git_workspace_fingerprint, repo_root=Path(context.working_directory)
        )
        calls: dict[str, tuple[int, ToolUseBlock]] = {}
        success: dict[str, tuple[int, str]] = {}
        last_write = -1
        for index, message in enumerate(payload.messages):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    calls[block.id] = (index, block)
                elif isinstance(block, ToolResultBlock):
                    call = calls.get(block.tool_use_id)
                    if call is None or not isinstance(block.content, dict):
                        continue
                    if call[1].name in {"write_file", "edit_file"} and not block.content.get("error"):
                        last_write = index
                    data = block.content.get("data")
                    if call[1].name == "bash" and isinstance(data, dict) and data.get("success") is True:
                        command = data.get("command")
                        fingerprint = data.get("workspace_fingerprint")
                        if isinstance(command, str) and isinstance(fingerprint, str):
                            success[command.strip()] = (index, fingerprint)
        missing = [
            command
            for command in self.commands
            if command.strip() not in success
            or success[command.strip()][0] <= last_write
            or success[command.strip()][1] != current
        ]
        return StopHookResult(
            blocking_reasons=(
                ["Configured validation commands have not all succeeded on the current workspace: " + ", ".join(missing)]
                if missing
                else []
            )
        )

