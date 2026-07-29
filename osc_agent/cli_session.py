"""管理本地 CLI 会话的启动、恢复和事件输出。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import typer

from osc_agent.application import AgentConversation, AgentInput, UserPrompt
from osc_agent.runtime.models import (
    AssistantDelta,
    ContextCompacted,
    ModelRequestStarted,
    ModelRetryScheduled,
    RunStopped,
    RuntimeEvent,
    ToolCompleted,
    ToolRequested,
)
from osc_agent.runtime.session_summary import SessionSummary, build_session_summary


def run_conversation(
    *,
    conversation: AgentConversation,
    repository_root: Path,
    initial_input: AgentInput | None,
    once: bool,
    quiet: bool,
) -> None:
    interactive = not once and sys.stdin.isatty()
    events = conversation.submit(initial_input)
    while True:
        stopped = False
        try:
            stopped = asyncio.run(_render(events, quiet=quiet))
        except KeyboardInterrupt:
            typer.echo("\nCancelled current turn.", err=True)
        _render_summary(conversation, repository_root)
        if not interactive:
            if stopped:
                raise typer.Exit(1)
            return
        try:
            prompt = typer.prompt("You", default="", show_default=False)
        except (KeyboardInterrupt, EOFError, typer.Abort):
            typer.echo()
            return
        prompt = prompt.strip()
        if prompt == "/exit":
            return
        if not prompt:
            continue
        events = conversation.submit(UserPrompt(text=prompt))


async def _render(
    events: AsyncIterator[RuntimeEvent],
    *,
    quiet: bool,
) -> bool:
    stopped = False
    tool_names: dict[str, str] = {}
    async for event in events:
        if isinstance(event, AssistantDelta):
            typer.echo(event.text, nl=False)
        elif isinstance(event, ModelRequestStarted):
            _status(f"→ Model round {event.round_number}", quiet)
        elif isinstance(event, ModelRetryScheduled):
            if event.had_partial_output:
                typer.echo()
            _status(
                f"↻ Model retry {event.attempt}/{event.max_attempts} "
                f"in {event.delay_seconds:.1f}s ({event.error_code})",
                quiet,
            )
        elif isinstance(event, ToolRequested):
            label = _tool_label(event.call.name, event.call.input)
            tool_names[event.call.id] = label
            _status(f"→ {label}", quiet)
        elif isinstance(event, ToolCompleted):
            label = tool_names.get(event.tool_use_id, "Tool")
            marker = "✗" if event.result.error is not None else "✓"
            _status(f"{marker} {label}", quiet)
        elif isinstance(event, ContextCompacted):
            _status(
                f"↘ Context compacted {event.before_messages}→{event.after_messages}",
                quiet,
            )
        elif isinstance(event, RunStopped):
            stopped = True
            typer.echo(f"\nStopped: {event.transition.reason}", err=True)
    typer.echo()
    return stopped


def _status(text: str, quiet: bool) -> None:
    if not quiet:
        typer.echo(text, err=True)


def _tool_label(name: str, input: dict) -> str:
    if name == "agent":
        agent = input.get("agent")
        return f"Agent {agent}" if isinstance(agent, str) else "Agent"
    if name == "skill":
        skill = input.get("skill")
        return f"Skill {skill}" if isinstance(skill, str) else "Skill"
    if name in {"read_file", "grep", "glob", "git_diff"}:
        path = input.get("path")
        if isinstance(path, str):
            return f"{name} {_bounded(path, 100)}"
    return name


def _bounded(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _render_summary(
    conversation: AgentConversation,
    repository_root: Path,
) -> None:
    snapshot = conversation.snapshot()
    if snapshot is None:
        return
    summary = build_session_summary(snapshot)
    typer.echo("\nSummary")
    typer.echo(f"  Session: {summary.session_id}")
    typer.echo(f"  Status: {summary.status}")
    typer.echo(f"  Working directory: {summary.working_directory}")
    if summary.worktree_path is not None:
        typer.echo(
            f"  Worktree: {summary.worktree_path} ({summary.worktree_branch})"
        )
    typer.echo(
        "  Touched files: "
        + (", ".join(summary.touched_files) if summary.touched_files else "not collected")
    )
    if summary.last_test_success is None:
        typer.echo("  Test: not collected")
    else:
        typer.echo(
            f"  Test: {'PASS' if summary.last_test_success else 'FAIL'} "
            "(test command recorded)"
        )
    typer.echo(
        f"  Verify: {summary.verification_verdict or 'not collected'}"
    )
    if summary.snapshot_complete is None:
        typer.echo("  Git snapshot: not collected")
    else:
        typer.echo(
            "  Git snapshot: "
            f"{'complete' if summary.snapshot_complete else 'truncated'}, "
            f"{len(summary.snapshot_files)} files"
        )
    if summary.reason:
        typer.echo(f"  Reason: {_bounded(summary.reason, 200)}")
    typer.echo(
        f'  Resume: osc-agent resume --repo "{repository_root}" {summary.session_id}'
    )


def render_session_summary(summary: SessionSummary, repository_root: Path) -> None:
    """Session CLI 复用与每轮结束相同的确定性摘要。"""

    typer.echo(f"Session: {summary.session_id}")
    typer.echo(f"Status: {summary.status}")
    typer.echo(f"Working directory: {summary.working_directory}")
    typer.echo(
        "Touched files: "
        + (", ".join(summary.touched_files) if summary.touched_files else "not collected")
    )
    typer.echo(
        "Test: "
        + (
            ("PASS" if summary.last_test_success else "FAIL")
            if summary.last_test_success is not None
            else "not collected"
        )
    )
    typer.echo(f"Verify: {summary.verification_verdict or 'not collected'}")
    typer.echo(
        f'RESUME: osc-agent resume --repo "{repository_root}" {summary.session_id}'
    )
