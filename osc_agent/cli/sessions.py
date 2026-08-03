"""查询仓库级 Session，并把 V6 Snapshot 投影为统一的只读 CLI 输出。"""

from __future__ import annotations

from pathlib import Path

import typer

from osc_agent.application.session_summary import SessionSummary, build_session_summary
from osc_agent.application.state_paths import ApplicationStatePaths
from osc_agent.runtime.messages import RuntimeMessage, TextBlock, ToolResultBlock, ToolUseBlock
from osc_agent.runtime.session_store import FileSessionStore


def session_store(repository_root: Path) -> FileSessionStore:
    return FileSessionStore(ApplicationStatePaths.for_repository(repository_root).sessions)


def latest_session_id(repository_root: Path) -> str | None:
    return session_store(repository_root).latest_session_id()


def list_sessions(repository_root: Path, *, limit: int) -> None:
    for item in session_store(repository_root).list_overviews(limit=limit):
        typer.echo(
            f"{item.session_id}\t{item.status}\t{item.updated_at}\t"
            f"{item.working_directory or '-'}"
        )


def show_session(repository_root: Path, session_id: str, *, messages: int) -> None:
    try:
        snapshot = session_store(repository_root).load(session_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if snapshot is None:
        raise typer.BadParameter(f"unknown session: {session_id}")
    render_session_summary(build_session_summary(snapshot), repository_root)
    state = snapshot.state
    typer.echo(f"Model: {snapshot.metadata.model}")
    allowed = state.capabilities.allowed_tools
    typer.echo(
        "Allowed tools: "
        + (", ".join(sorted(allowed)) if allowed is not None else "all registered tools")
    )
    typer.echo(f"Plan: {state.permissions.plan_path or '-'}")
    typer.echo(
        "Completion evidence required: "
        + (", ".join(sorted(state.completion_requirements.required_evidence)) or "none")
    )
    grant_tools = sorted({grant.tool_name for grant in state.permissions.grants})
    typer.echo(
        f"Session permissions: {len(state.permissions.grants)}"
        + (f" ({', '.join(grant_tools)})" if grant_tools else "")
    )
    if messages:
        typer.echo("Recent messages:")
        for message in snapshot.messages[-messages:]:
            typer.echo(f"  {message.role}: {_message_preview(message)}")


def render_session_summary(
    summary: SessionSummary,
    repository_root: Path,
    *,
    heading: str | None = None,
) -> None:
    prefix = "  " if heading is not None else ""
    if heading is not None:
        typer.echo(f"\n{heading}")
    typer.echo(f"{prefix}Session: {summary.session_id}")
    typer.echo(f"{prefix}Status: {summary.status}")
    typer.echo(f"{prefix}Working directory: {summary.working_directory}")
    if summary.worktree_path is not None:
        typer.echo(
            f"{prefix}Worktree: {summary.worktree_path} ({summary.worktree_branch})"
        )
    typer.echo(
        f"{prefix}Touched files: "
        + (", ".join(summary.touched_files) if summary.touched_files else "not collected")
    )
    test = "not collected"
    if summary.last_test_success is not None:
        test = (
            "PASS (test command recorded)"
            if summary.last_test_success
            else "FAIL (test command recorded)"
        )
    typer.echo(f"{prefix}Test: {test}")
    typer.echo(
        f"{prefix}Verify: {summary.verification_verdict or 'not collected'}"
    )
    if summary.snapshot_complete is not None:
        typer.echo(
            f"{prefix}Git snapshot: "
            f"{'complete' if summary.snapshot_complete else 'truncated'}, "
            f"{len(summary.snapshot_files)} files"
        )
    else:
        typer.echo(f"{prefix}Git snapshot: not collected")
    if summary.reason:
        typer.echo(f"{prefix}Reason: {_bounded(summary.reason, 200)}")
    typer.echo(
        f'{prefix}Resume: osc-agent resume --repo "{repository_root}" '
        f"{summary.session_id}"
    )


def _message_preview(message: RuntimeMessage) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text = " ".join(block.text.split())
            parts.append(text[:500] + ("…" if len(text) > 500 else ""))
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[tool_use {block.name}]")
        elif isinstance(block, ToolResultBlock):
            parts.append(f"[tool_result {block.tool_use_id} {'error' if block.is_error else 'ok'}]")
    return " ".join(parts)


def _bounded(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"
