"""实施 Bot 工具权限、仓库约束和完成验证策略。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import asyncio
from collections.abc import Callable

from pydantic import JsonValue

from osc_agent.bot.models import RepositoryBotConfig
from osc_agent.runtime.hooks import HookBlock, HookContinue, PreToolUsePayload, StopHookPayload, StopHookResult
from osc_agent.runtime.models import Allow, ContractModel, Deny, PermissionDecision, ToolResultBlock, ToolUseBlock, ToolUseContext
from osc_agent.tools.git import git_snapshot, git_workspace_fingerprint
from osc_agent.runtime.permissions import PermissionPolicy
from osc_agent.runtime.tool import Tool


class BotPermissionPolicy(PermissionPolicy):
    """`/osa implement` 只授权仓库写入和隔离进程，不授权外部或破坏性动作。"""

    def __init__(self, *, implementation_approved: bool) -> None:
        self.implementation_approved = implementation_approved

    async def decide(self, tool: Tool[ContractModel, ContractModel], input: ContractModel, context: ToolUseContext) -> PermissionDecision:
        if not context.capabilities.permits_tool(tool.name):
            return Deny(reason=f"tool {tool.name} is outside the bot capability scope")
        if context.permission_mode == "plan" and not tool.is_read_only(input) and tool.name != "write_plan":
            return Deny(reason=f"tool {tool.name} is not allowed in plan mode")
        risk = tool.permission_risk(input)
        if tool.is_destructive(input):
            if self.implementation_approved and risk in {"write", "process"}:
                return Allow(updated_input=input.model_dump(mode="json"))
            return Deny(reason=f"bot policy denies unapproved {risk} tool call {tool.name}")
        updated: dict[str, JsonValue] = input.model_dump(mode="json")
        return Allow(updated_input=updated)


class BotRepositoryPolicyHook:
    def __init__(
        self,
        config: RepositoryBotConfig,
        on_violation: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.on_violation = on_violation

    def _block(self, reason: str) -> HookBlock:
        if self.on_violation is not None:
            self.on_violation(reason)
        return HookBlock(reason=reason)

    async def __call__(self, payload: PreToolUsePayload, context: ToolUseContext) -> HookContinue | HookBlock:
        if payload.tool_name not in {"write_file", "edit_file"}:
            return HookContinue()
        value = payload.input.get("path")
        if not isinstance(value, str):
            return self._block("bot write tool requires a repository-relative path")
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or PureWindowsPath(value).is_absolute()
            or ".." in path.parts
        ):
            return self._block("bot write path escapes the repository")
        normalized = path.as_posix()
        path = PurePosixPath(normalized)
        if any(_matches(path, pattern) for pattern in self.config.denied_paths):
            return self._block(f"bot repository policy protects path: {normalized}")
        try:
            snapshot = await asyncio.to_thread(
                git_snapshot,
                repo_root=Path(context.working_directory),
            )
        except (OSError, ValueError) as exc:
            return self._block(f"bot cannot verify repository change limits: {exc}")
        files = {str(item["path"]).replace("\\", "/") for item in snapshot["files"]}
        patch_bytes = len(str(snapshot["patch"]).encode("utf-8"))
        if len(files) >= self.config.max_changed_files and normalized not in files:
            return self._block("bot repository has reached the changed-file limit")
        if patch_bytes >= self.config.max_patch_bytes:
            return self._block("bot repository has reached the patch-size limit")
        return HookContinue()


def _matches(path: PurePosixPath, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    if normalized.endswith("/**"):
        prefix = normalized[:-3].rstrip("/")
        return path.as_posix() == prefix or path.as_posix().startswith(prefix + "/")
    return path.match(normalized)


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
