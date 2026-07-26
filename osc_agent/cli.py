from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from pydantic import JsonValue

from osc_agent.application import (
    ApplicationServices,
    build_application,
    build_session_store,
    build_skill_catalog,
)
from osc_agent.cli_session import render_session_summary, run_conversation
from osc_agent.config import load_settings
from osc_agent.doctor import run_doctor
from osc_agent.runtime.models import (
    ApprovalResponse,
    Ask,
    CapabilityScope,
    ResumeQueryParams,
    RuntimeMessage,
    StartQueryParams,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from osc_agent.runtime.session_summary import build_session_summary
from osc_agent.skills.models import SkillInvocation


app = typer.Typer(help="Claude Code style extensible coding agent.")
skill_app = typer.Typer(help="List and run validated Skills.")
session_app = typer.Typer(help="Inspect repository-scoped Sessions.")
app.add_typer(skill_app, name="skill")
app.add_typer(session_app, name="session")

RepoOption = Annotated[Path, typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True)]


@app.command("run")
def run_agent(
    repo: RepoOption,
    prompt: Annotated[str | None, typer.Argument(help="Task for the general coding agent.")] = None,
    once: Annotated[bool, typer.Option("--once", help="Run one turn even in a TTY.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Hide compact progress events.")] = False,
) -> None:
    """Start a new general AgentRuntime Session."""

    task = prompt
    if not task:
        if not sys.stdin.isatty():
            raise typer.BadParameter("task is required when stdin is not a TTY")
        task = typer.prompt("Task")
    session_id = str(uuid4())
    services = _application(repo)
    settings = load_settings()

    typer.echo(f"Session: {session_id}")
    run_conversation(
        services=services,
        session_id=session_id,
        repository_root=repo,
        initial_events=services.runtime.query(
            StartQueryParams(
                session_id=session_id,
                model=settings.model_id or "",
                system_prompt=(
                    "Use repository evidence and the smallest safe change that satisfies the task.\n\n"
                    + services.discovery_prompt
                ),
                messages=[RuntimeMessage(role="user", content=[TextBlock(text=task)])],
                repository_root=str(repo),
                capabilities=services.general_capabilities,
                config=services.query_config,
            )
        ),
        once=once,
        quiet=quiet,
    )


@app.command("resume")
def resume_session(
    repo: RepoOption,
    session_id: Annotated[str | None, typer.Argument(help="Existing Session ID.")] = None,
    prompt: Annotated[str | None, typer.Option("--prompt", help="Optional new user message.")] = None,
    latest: Annotated[bool, typer.Option("--latest", help="Resume the latest valid Session.")] = False,
    once: Annotated[bool, typer.Option("--once", help="Run one turn even in a TTY.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", help="Hide compact progress events.")] = False,
) -> None:
    """Resume a Session transcript without contribution-stage knowledge."""

    if (session_id is None) == (not latest):
        raise typer.BadParameter("provide exactly one of SESSION_ID or --latest")
    if latest:
        session_id = build_session_store(repo).latest_session_id()
        if session_id is None:
            raise typer.BadParameter("no valid Session exists for this repository")
    assert session_id is not None
    services = _application(repo)
    messages = [RuntimeMessage(role="user", content=[TextBlock(text=prompt)])] if prompt else []

    typer.echo(f"Session: {session_id}")
    run_conversation(
        services=services,
        session_id=session_id,
        repository_root=repo,
        initial_events=services.runtime.query(
            ResumeQueryParams(
                session_id=session_id,
                messages=messages,
                repository_root=str(repo),
                config=services.query_config,
            )
        ),
        once=once,
        quiet=quiet,
    )


@app.command("contribute")
def contribute(
    repo: RepoOption,
    repo_url: Annotated[str, typer.Option("--repo-url")],
    goal: Annotated[str | None, typer.Option("--goal")] = None,
    once: Annotated[bool, typer.Option("--once")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    """Invoke the single open-source-contribution Skill."""

    _run_inline_skill(
        "open-source-contribution",
        repo,
        {"repo_url": repo_url, "goal": goal},
        once=once,
        quiet=quiet,
    )


@app.command("doctor")
def doctor(
    repo: RepoOption = Path.cwd(),
    local_only: Annotated[
        bool,
        typer.Option("--local-only", help="Skip the live model connectivity probe."),
    ] = False,
) -> None:
    """Check required local dependencies, configuration, state, and model access."""

    results = asyncio.run(
        run_doctor(
            repository_root=repo,
            settings=load_settings(),
            local_only=local_only,
        )
    )
    for item in results:
        typer.echo(f"{item.status}\t{item.name}\t{item.message}")
    if any(item.status == "FAIL" for item in results):
        raise typer.Exit(1)


@skill_app.command("list")
def list_skills(repo: RepoOption = Path.cwd()) -> None:
    """List frontmatter metadata without loading Skill bodies or resources."""

    catalog = build_skill_catalog(repo)
    for descriptor in catalog.list():
        typer.echo(f"{descriptor.manifest.name}\t{descriptor.source}\t{descriptor.manifest.description}")
    for descriptor in catalog.blocked_overrides():
        typer.echo(
            f"WARN\t{descriptor.manifest.name}\t{descriptor.source} override blocked: built-in name is reserved",
            err=True,
        )


@skill_app.command("run")
def run_skill(
    name: Annotated[str, typer.Argument(help="Exact Skill name.")],
    repo: RepoOption,
    arguments: Annotated[str, typer.Option("--arguments", help="Strict JSON object.")] = "{}",
    once: Annotated[bool, typer.Option("--once")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
) -> None:
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--arguments must be valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("--arguments must decode to an object")
    descriptor = build_skill_catalog(repo).get(name)
    if descriptor is None:
        raise typer.BadParameter(f"unknown skill: {name}")
    if descriptor.manifest.context == "inline":
        _run_inline_skill(name, repo, payload, once=once, quiet=quiet)
        return

    services = _application(repo)
    result = asyncio.run(
        services.skill_executor.execute(
            SkillInvocation(
                name=name,
                arguments=payload,
                session_id=str(uuid4()),
                working_directory=str(repo),
                caller_capabilities=CapabilityScope(),
                trigger="user",
            )
        )
    )
    typer.echo(result.model_dump_json(indent=2))
    if result.status == "failed":
        raise typer.Exit(1)


def _run_inline_skill(
    name: str,
    repo: Path,
    arguments: dict[str, JsonValue],
    *,
    once: bool,
    quiet: bool,
) -> None:
    services = _application(repo)
    session_id = str(uuid4())
    typer.echo(f"Session: {session_id}")

    try:
        run_conversation(
            services=services,
            session_id=session_id,
            repository_root=repo,
            initial_events=services.skill_command_runner.run(
                name=name,
                arguments=arguments,
                session_id=session_id,
                working_directory=str(repo),
            ),
            once=once,
            quiet=quiet,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _application(repo: Path) -> ApplicationServices:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise typer.BadParameter("ANTHROPIC_API_KEY is required for model execution")
    if not settings.model_id:
        raise typer.BadParameter("MODEL_ID is required for model execution")
    return build_application(
        settings=settings,
        repo_root=repo,
        approval_handler=_approve,
        question_handler=_ask_questions,
    )


async def _approve(decision: Ask) -> ApprovalResponse:
    typer.echo(f"\nTool: {decision.tool_name}")
    typer.echo(f"Working directory: {decision.working_directory}")
    typer.echo(f"Risk: {decision.risk}")
    typer.echo("Input preview:")
    typer.echo(json.dumps(decision.preview, ensure_ascii=False, indent=2))
    if decision.tool_name == "powershell":
        typer.echo(
            "WARNING: this command runs on the Host without an OS sandbox and may access files available to the current Windows user."
        )
    if decision.risk in {"write", "process"}:
        typer.echo("  1. Allow once")
        typer.echo("  2. Allow for this Session")
        typer.echo("  3. Deny")
        choice = typer.prompt("Choose", type=int)
        mapping = {
            1: "allow_once",
            2: "allow_for_session",
            3: "deny",
        }
        if choice not in mapping:
            raise typer.BadParameter("permission choice must be 1, 2, or 3")
        return ApprovalResponse(choice=mapping[choice])
    return ApprovalResponse(
        choice="allow_once" if typer.confirm(decision.prompt, default=False) else "deny"
    )


async def _ask_questions(questions: list[dict[str, JsonValue]]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for question in questions:
        question_id = str(question["id"])
        text = str(question["question"])
        options = question.get("options", [])
        typer.echo(f"\n{text}")
        labels: list[str] = []
        option_ids: list[str] = []
        for index, option in enumerate(options, start=1):
            if isinstance(option, dict):
                label = str(option.get("label", ""))
                labels.append(label)
                option_ids.append(str(option.get("id", "")))
                typer.echo(f"  {index}. {label} — {option.get('description', '')}")
        raw = typer.prompt("Choose a number or enter a custom answer")
        try:
            selected = int(raw)
        except ValueError:
            answers[question_id] = raw
        else:
            if selected < 1 or selected > len(labels):
                raise typer.BadParameter("question choice is outside the available options")
            answers[question_id] = option_ids[selected - 1]
    return answers


@session_app.command("list")
def list_sessions(
    repo: RepoOption = Path.cwd(),
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
) -> None:
    """List Sessions isolated to the selected repository."""

    for item in build_session_store(repo).list_overviews(limit=limit):
        typer.echo(
            f"{item.session_id}\t{item.status}\t{item.updated_at}\t"
            f"{item.working_directory or '-'}"
        )


@session_app.command("show")
def show_session(
    session_id: Annotated[str, typer.Argument()],
    repo: RepoOption = Path.cwd(),
    messages: Annotated[int, typer.Option("--messages", min=0)] = 0,
) -> None:
    """Show Session metadata without exposing transcript content by default."""

    try:
        snapshot = build_session_store(repo).load(session_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if snapshot is None:
        raise typer.BadParameter(f"unknown session: {session_id}")
    render_session_summary(build_session_summary(snapshot), repo)
    typer.echo(f"Model: {snapshot.metadata.model}")
    allowed = snapshot.runtime_state.capabilities or snapshot.metadata.capabilities
    typer.echo(
        "Allowed tools: "
        + (
            ", ".join(sorted(allowed.allowed_tools))
            if allowed.allowed_tools is not None
            else "all registered tools"
        )
    )
    typer.echo(f"Plan: {snapshot.runtime_state.plan_path or '-'}")
    requirements = snapshot.metadata.completion_requirements
    if snapshot.runtime_state.completion_requirements is not None:
        requirements = requirements.tighten(
            snapshot.runtime_state.completion_requirements
        )
    typer.echo(
        "Completion evidence required: "
        + (
            ", ".join(sorted(requirements.required_evidence))
            or "none"
        )
    )
    grant_tools = sorted(
        {grant.tool_name for grant in snapshot.runtime_state.permission_grants}
    )
    typer.echo(
        f"Session permissions: {len(snapshot.runtime_state.permission_grants)}"
        + (f" ({', '.join(grant_tools)})" if grant_tools else "")
    )
    if messages:
        typer.echo("Recent messages:")
        for message in snapshot.messages[-messages:]:
            typer.echo(f"  {message.role}: {_message_preview(message)}")


def _message_preview(message: RuntimeMessage) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text = " ".join(block.text.split())
            parts.append(text[:500] + ("…" if len(text) > 500 else ""))
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[tool_use {block.name}]")
        elif isinstance(block, ToolResultBlock):
            parts.append(
                f"[tool_result {block.tool_use_id} "
                f"{'error' if block.is_error else 'ok'}]"
            )
    return " ".join(parts)


if __name__ == "__main__":
    app()
