"""构建模型上下文并管理会话记录和大型工具结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import Field

from osc_agent.runtime.models import (
    ContractModel,
    QueryConfig,
    RuntimeMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseContext,
)
from osc_agent.runtime.gateway import ModelCompleted, ModelGateway, ModelRequest
from osc_agent.runtime.instructions import RepositoryInstructionResolver
from osc_agent.runtime.session_store import ToolResultStore


class MemoryToolResultStore:
    """测试和未配置持久层时的进程内保底实现。"""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}

    def persist(self, *, session_id: str, tool_use_id: str, content: str) -> str:
        result_id = tool_use_id
        self._values[(session_id, result_id)] = content
        return result_id

    def read(self, *, session_id: str, result_id: str) -> str:
        try:
            return self._values[(session_id, result_id)]
        except KeyError as exc:
            raise ValueError("unknown tool result for this session") from exc


class SessionTranscript(ContractModel):
    """会话的权威记录；Context Pipeline 只能读取，不能覆盖。"""

    session_id: str = Field(min_length=1)
    messages: list[RuntimeMessage] = Field(default_factory=list)

    def append(self, message: RuntimeMessage) -> None:
        # 重新赋值以触发 Pydantic 的 assignment validation，避免 list 原地修改绕过契约。
        self.messages = [*self.messages, message]

    def snapshot(self) -> list[RuntimeMessage]:
        return [message.model_copy(deep=True) for message in self.messages]


class ContextProjection(ContractModel):
    messages: list[RuntimeMessage] = Field(min_length=1)
    source_message_count: int = Field(ge=1)
    compacted: bool = False
    reason: str | None = None
    summary_input_tokens: int = Field(default=0, ge=0)
    summary_output_tokens: int = Field(default=0, ge=0)
    system_reminder: str = ""


class ContextSummary(ContractModel):
    text: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class ContextSummarizer(Protocol):
    async def summarize(self, messages: list[RuntimeMessage], *, reason: str) -> ContextSummary: ...


class DeterministicContextSummarizer:
    """无额外模型调用的保底摘要器；生产环境可通过依赖注入替换。"""

    async def summarize(self, messages: list[RuntimeMessage], *, reason: str) -> ContextSummary:
        user_text = [
            block.text
            for message in messages
            if message.role == "user"
            for block in message.content
            if isinstance(block, TextBlock)
        ]
        recent = user_text[-1][:1_000] if user_text else "(none)"
        return ContextSummary(
            text=(
                f"[Context compacted: {reason}]\n"
                f"Messages summarized: {len(messages)}\n"
                f"Most recent user context:\n{recent}"
            )
        )


class GatewayContextSummarizer:
    """使用同一模型边界生成面向继续执行的压缩摘要。"""

    def __init__(self, gateway: ModelGateway, *, model: str, max_output_tokens: int = 2_048) -> None:
        self.gateway = gateway
        self.model = model
        self.max_output_tokens = max_output_tokens

    async def summarize(self, messages: list[RuntimeMessage], *, reason: str) -> ContextSummary:
        completed: ModelCompleted | None = None
        async for event in self.gateway.stream(
            ModelRequest(
                model=self.model,
                system_prompt=(
                    "Summarize the coding-agent transcript for exact continuation. Preserve the current user "
                    "goal, confirmed choices, current plan, modified files, commands and test results, unresolved "
                    "errors, and the next action. Do not invent progress. Return plain text only."
                ),
                messages=messages,
                tools=[],
                max_output_tokens=self.max_output_tokens,
            )
        ):
            if isinstance(event, ModelCompleted):
                completed = event
        if completed is None:
            raise ValueError("context summarizer returned no completed message")
        text = "\n".join(block.text for block in completed.message.content if isinstance(block, TextBlock)).strip()
        if not text:
            raise ValueError("context summarizer returned empty text")
        return ContextSummary(
            text=text,
            input_tokens=completed.input_tokens,
            output_tokens=completed.output_tokens,
        )


class ContextPipeline:
    def __init__(
        self,
        *,
        summarizer: ContextSummarizer | None = None,
        tool_result_store: ToolResultStore | None = None,
        instruction_resolver: RepositoryInstructionResolver | None = None,
    ) -> None:
        self.summarizer = summarizer or DeterministicContextSummarizer()
        self.tool_result_store = tool_result_store or MemoryToolResultStore()
        self.instruction_resolver = instruction_resolver or RepositoryInstructionResolver()

    async def project(
        self,
        transcript: SessionTranscript,
        *,
        config: QueryConfig,
        working_directory: str,
        runtime_context: ToolUseContext | None = None,
        force_reason: str | None = None,
    ) -> ContextProjection:
        messages = transcript.snapshot()
        reasons: list[str] = []

        if self._enforce_tool_result_budget(
            messages,
            max_chars=config.max_tool_result_chars,
            session_id=transcript.session_id,
        ):
            reasons.append("tool_result_budget")
        if self._micro_compact(messages, keep_recent=config.keep_recent_tool_results):
            reasons.append("micro_compact")

        should_compact = force_reason is not None or _estimate_chars(messages) > config.auto_compact_chars
        if should_compact:
            reason = force_reason or "auto_compact"
            messages, summary = await self._compact(messages, reason=reason)
            reasons.append(reason)
        else:
            summary = ContextSummary(text="No summary generated")

        reminder = _runtime_reminder(runtime_context, self.instruction_resolver)

        return ContextProjection(
            messages=messages,
            source_message_count=len(transcript.messages),
            compacted=bool(reasons),
            reason=", ".join(reasons) or None,
            summary_input_tokens=summary.input_tokens if should_compact else 0,
            summary_output_tokens=summary.output_tokens if should_compact else 0,
            system_reminder=reminder,
        )

    def _enforce_tool_result_budget(
        self,
        messages: list[RuntimeMessage],
        *,
        max_chars: int,
        session_id: str,
    ) -> bool:
        results = _tool_results(messages)
        total = sum(len(_json_text(block.content)) for block in results)
        changed = False
        for block in sorted(results, key=lambda item: len(_json_text(item.content)), reverse=True):
            if total <= max_chars:
                break
            original = _json_text(block.content)
            path = self.tool_result_store.persist(
                session_id=session_id,
                tool_use_id=block.tool_use_id,
                content=original,
            )
            preview = original[:1_000]
            block.content = f"[Tool result persisted as {path}; use read_tool_result]\nPreview:\n{preview}"
            total = sum(len(_json_text(item.content)) for item in results)
            changed = True
        return changed

    @staticmethod
    def _micro_compact(messages: list[RuntimeMessage], *, keep_recent: int) -> bool:
        results = _tool_results(messages)
        if len(results) <= keep_recent:
            return False
        changed = False
        older = results if keep_recent == 0 else results[:-keep_recent]
        for block in older:
            content = _json_text(block.content)
            if len(content) > 200 and not content.startswith("[Tool result persisted"):
                block.content = f"[Earlier tool result compacted. Preview: {content[:200]}]"
                changed = True
        return changed

    async def _compact(
        self,
        messages: list[RuntimeMessage],
        *,
        reason: str,
    ) -> tuple[list[RuntimeMessage], ContextSummary]:
        groups = _group_by_api_round(messages)
        preserve_count = min(2, len(groups))
        if len(groups) <= preserve_count:
            preserve_count = max(0, len(groups) - 1)
        summarized_groups = groups if preserve_count == 0 else groups[:-preserve_count]
        preserved_groups = [] if preserve_count == 0 else groups[-preserve_count:]
        summarized = [message for group in summarized_groups for message in group]
        try:
            summary = await self.summarizer.summarize(summarized, reason=reason)
        except Exception:  # noqa: BLE001 - 压缩失败必须回退，不能破坏权威 transcript。
            summary = await DeterministicContextSummarizer().summarize(summarized, reason=reason)
        projection = [RuntimeMessage(role="user", content=[TextBlock(text=summary.text)])]
        projection.extend(message for group in preserved_groups for message in group)
        return projection, summary


def _runtime_reminder(
    context: ToolUseContext | None,
    instruction_resolver: RepositoryInstructionResolver | None = None,
) -> str:
    if context is None:
        return ""
    lines = [
        "<session_runtime>",
        f"permission_mode: {context.permission_mode}",
        f"working_directory: {context.working_directory}",
    ]
    if context.plan_path:
        plan = (Path(context.state_directory) / "plans" / context.plan_path).resolve()
        plans_root = (Path(context.state_directory) / "plans").resolve()
        if plan.parent == plans_root and plan.is_file():
            lines.extend(["current_plan:", plan.read_text(encoding="utf-8")])
    if context.worktree is not None:
        lines.extend(
            [
                f"worktree_path: {context.worktree.path}",
                f"worktree_branch: {context.worktree.branch}",
                f"worktree_base_commit: {context.worktree.base_commit}",
            ]
        )
    documents = (instruction_resolver or RepositoryInstructionResolver()).load(
        Path(context.working_directory),
        context.instruction_state,
    )
    if documents:
        lines.append("<repository_instructions>")
        lines.append(
            "Files at the same scope have equal priority. If AGENTS.md and CLAUDE.md contain a "
            "task-relevant conflict, ask the user before mutating files."
        )
        for document in documents:
            lines.extend(
                [
                    f'<instruction path="{document.path}" kind="{document.kind}" '
                    f'scope="{document.scope_directory}" hash="{document.content_hash}">',
                    document.content,
                    "</instruction>",
                ]
            )
        lines.append("</repository_instructions>")
    lines.append("</session_runtime>")
    return "\n".join(lines)


def _group_by_api_round(messages: list[RuntimeMessage]) -> list[list[RuntimeMessage]]:
    """按 assistant 回合切分，确保 tool_use 与其后 tool_result 留在同一组。"""

    groups: list[list[RuntimeMessage]] = []
    current: list[RuntimeMessage] = []
    for message in messages:
        if message.role == "assistant" and current:
            groups.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        groups.append(current)
    return groups


def _tool_results(messages: list[RuntimeMessage]) -> list[ToolResultBlock]:
    return [
        block
        for message in messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _estimate_chars(messages: list[RuntimeMessage]) -> int:
    return len(json.dumps([message.model_dump(mode="json") for message in messages], ensure_ascii=False))
