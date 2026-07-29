"""实施 Bot Worker 的工具权限与仓库约束。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import asyncio
from collections.abc import Callable

from pydantic import JsonValue

from osc_agent.bot.domain.repositories import RepositoryBotConfig
from osc_agent.runtime.hooks import HookBlock, HookContinue, PreToolUsePayload
from osc_agent.contracts import ContractModel
from osc_agent.runtime.tool_models import Allow, Deny, PermissionDecision, ToolUseContext
from osc_agent.workspaces.git_state import git_snapshot
from osc_agent.workspaces.path_policy import repo_path_matches
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
        if any(repo_path_matches(normalized, pattern) for pattern in self.config.denied_paths):
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


