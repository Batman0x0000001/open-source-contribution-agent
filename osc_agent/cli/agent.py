"""组装本地 Agent，并完整处理交互授权、多轮提交和 Runtime 事件展示。"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import typer
from pydantic import JsonValue

from osc_agent.application import (
    AgentApplication,
    AgentApplicationConfig,
    AgentConversation,
    AgentProfile,
    UserPrompt,
    build_agent_application,
)
from osc_agent.application.agent import ApprovalHandler, QuestionHandler
from osc_agent.application.session_summary import build_session_summary
from osc_agent.configuration import load_agent_settings
from osc_agent.runtime.events import (
    AssistantDelta,
    ContextCompacted,
    ModelRequestStarted,
    ModelRetryScheduled,
    RunStopped,
    RuntimeEvent,
    ToolCompleted,
    ToolRequested,
)
from osc_agent.runtime.tool_models import ApprovalResponse, Ask


def build_general_profile() -> AgentProfile:
    return AgentProfile(
        profile_id="local_debug",
        system_prompt="Use repository evidence and the smallest safe change that satisfies the task.",
    )


def build_skill_profile(name: str) -> AgentProfile:
    return AgentProfile(
        profile_id="local_debug",
        system_prompt="Use repository evidence and follow the explicitly invoked Skill.",
        allowed_initial_skills=frozenset({name}),
    )


def build_cli_application(
    repository_root: Path,
    profile: AgentProfile,
    *,
    approval_handler: ApprovalHandler | None = None,
    question_handler: QuestionHandler | None = None,
) -> AgentApplication:
    return build_agent_application(
        AgentApplicationConfig(
            settings=load_agent_settings(),
            repository_root=repository_root,
            profile=profile,
            approval_handler=approval_handler or approve_tool,
            question_handler=question_handler or ask_questions,
        )
    )


def run_conversation(
    *,
    conversation: AgentConversation,
    repository_root: Path,
    initial_events: AsyncIterator[RuntimeEvent],
    once: bool,
    quiet: bool,
) -> None:
    interactive = not once and sys.stdin.isatty()
    events = initial_events
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
        if prompt:
            events = conversation.resume(UserPrompt(text=prompt))


async def approve_tool(decision: Ask) -> ApprovalResponse:
    typer.echo(f"\nTool: {decision.tool_name}")
    typer.echo(f"Working directory: {decision.working_directory}")
    typer.echo(f"Risk: {decision.risk}")
    typer.echo("Input preview:")
    typer.echo(json.dumps(decision.preview, ensure_ascii=False, indent=2))
    if decision.tool_name == "bash":
        typer.echo("WARNING: this Bash command runs on the Host without an OS sandbox.")
    if decision.risk in {"write", "process"}:
        typer.echo("  1. Allow once\n  2. Allow for this Session\n  3. Deny")
        choice = typer.prompt("Choose", type=int)
        mapping = {1: "allow_once", 2: "allow_for_session", 3: "deny"}
        if choice not in mapping:
            raise typer.BadParameter("permission choice must be 1, 2, or 3")
        return ApprovalResponse(choice=mapping[choice])
    return ApprovalResponse(
        choice="allow_once" if typer.confirm(decision.prompt, default=False) else "deny"
    )


async def ask_questions(questions: list[dict[str, JsonValue]]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for question in questions:
        question_id = str(question["id"])
        typer.echo(f"\n{question['question']}")
        option_ids: list[str] = []
        for index, option in enumerate(question.get("options", []), start=1):
            if isinstance(option, dict):
                option_ids.append(str(option.get("id", "")))
                typer.echo(
                    f"  {index}. {option.get('label', '')} — {option.get('description', '')}"
                )
        raw = typer.prompt("Choose a number or enter a custom answer")
        try:
            selected = int(raw)
        except ValueError:
            answers[question_id] = raw
        else:
            if selected < 1 or selected > len(option_ids):
                raise typer.BadParameter("question choice is outside the available options")
            answers[question_id] = option_ids[selected - 1]
    return answers


async def _render(events: AsyncIterator[RuntimeEvent], *, quiet: bool) -> bool:
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
            _status(f"{'✗' if event.result.error is not None else '✓'} {label}", quiet)
        elif isinstance(event, ContextCompacted):
            _status(f"↘ Context compacted {event.before_messages}→{event.after_messages}", quiet)
        elif isinstance(event, RunStopped):
            stopped = True
            typer.echo(f"\nStopped: {event.transition.reason}", err=True)
    typer.echo()
    return stopped


def _render_summary(conversation: AgentConversation, repository_root: Path) -> None:
    snapshot = conversation.snapshot()
    if snapshot is None:
        return
    summary = build_session_summary(snapshot)
    typer.echo("\nSummary")
    typer.echo(f"  Session: {summary.session_id}")
    typer.echo(f"  Status: {summary.status}")
    typer.echo(f"  Working directory: {summary.working_directory}")
    if summary.worktree_path is not None:
        typer.echo(f"  Worktree: {summary.worktree_path} ({summary.worktree_branch})")
    typer.echo(
        "  Touched files: "
        + (", ".join(summary.touched_files) if summary.touched_files else "not collected")
    )
    test = "not collected"
    if summary.last_test_success is not None:
        test = "PASS (test command recorded)" if summary.last_test_success else "FAIL (test command recorded)"
    typer.echo(f"  Test: {test}")
    typer.echo(f"  Verify: {summary.verification_verdict or 'not collected'}")
    if summary.snapshot_complete is not None:
        typer.echo(
            "  Git snapshot: "
            f"{'complete' if summary.snapshot_complete else 'truncated'}, "
            f"{len(summary.snapshot_files)} files"
        )
    else:
        typer.echo("  Git snapshot: not collected")
    if summary.reason:
        typer.echo(f"  Reason: {_bounded(summary.reason, 200)}")
    typer.echo(f'  Resume: osc-agent resume --repo "{repository_root}" {summary.session_id}')


def _status(text: str, quiet: bool) -> None:
    if not quiet:
        typer.echo(text, err=True)


def _tool_label(name: str, input: dict) -> str:
    if name in {"agent", "skill"}:
        value = input.get(name)
        return f"{name.title()} {value}" if isinstance(value, str) else name.title()
    if name in {"read_file", "grep", "glob", "git_diff"} and isinstance(input.get("path"), str):
        return f"{name} {_bounded(input['path'], 100)}"
    return name


def _bounded(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"
