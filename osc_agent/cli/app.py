"""定义本地 osc-agent 命令行入口及其子命令。"""

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
    SkillInput,
    UserPrompt,
)
from osc_agent.cli.agent import (
    approve_tool,
    ask_questions,
    build_cli_application,
    build_general_profile,
    build_skill_profile,
    run_conversation,
)
from osc_agent.cli.sessions import (
    latest_session_id,
    list_sessions as render_session_list,
    show_session as render_session,
)
from osc_agent.configuration import load_agent_settings
from osc_agent.cli.doctor import run_doctor
from osc_agent.skills.catalog import build_skill_catalog


app = typer.Typer(help="Extensible coding agent for open-source contribution workflows.")
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
    agent_application = build_cli_application(
        repo,
        build_general_profile(),
        approval_handler=approve_tool,
        question_handler=ask_questions,
    )
    conversation = agent_application.open_session(session_id)

    typer.echo(f"Session: {session_id}")
    run_conversation(
        conversation=conversation,
        repository_root=repo,
        initial_input=UserPrompt(text=task),
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
        session_id = latest_session_id(repo)
        if session_id is None:
            raise typer.BadParameter("no valid Session exists for this repository")
    assert session_id is not None
    agent_application = build_cli_application(
        repo,
        build_general_profile(resume=True),
        approval_handler=approve_tool,
        question_handler=ask_questions,
    )
    conversation = agent_application.open_session(session_id)
    typer.echo(f"Session: {session_id}")
    run_conversation(
        conversation=conversation,
        repository_root=repo,
        initial_input=UserPrompt(text=prompt) if prompt else None,
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
            settings=load_agent_settings(),
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
        typer.echo(
            f"{descriptor.manifest.name}\t{descriptor.source}\t"
            f"{descriptor.manifest.description}\t{descriptor.manifest.when_to_use}"
        )
    for diagnostic in catalog.diagnostics():
        typer.echo(
            f"WARN\t{diagnostic.skill_name or '-'}\t{diagnostic.source}\t"
            f"{diagnostic.code}: {diagnostic.message} ({diagnostic.path})",
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
    catalog = build_skill_catalog(repo)
    descriptor = catalog.get(name)
    if descriptor is None:
        raise typer.BadParameter(f"unknown skill: {name}")
    if descriptor not in catalog.list_user_invocable():
        raise typer.BadParameter("skill is not user invocable")
    _run_inline_skill(name, repo, payload, once=once, quiet=quiet)


def _run_inline_skill(
    name: str,
    repo: Path,
    arguments: dict[str, JsonValue],
    *,
    once: bool,
    quiet: bool,
) -> None:
    session_id = str(uuid4())
    agent_application = build_cli_application(
        repo,
        build_skill_profile(name),
        approval_handler=approve_tool,
        question_handler=ask_questions,
    )
    conversation = agent_application.open_session(session_id)
    typer.echo(f"Session: {session_id}")

    try:
        run_conversation(
            conversation=conversation,
            repository_root=repo,
            initial_input=SkillInput(name=name, arguments=arguments),
            once=once,
            quiet=quiet,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@session_app.command("list")
def list_sessions(
    repo: RepoOption = Path.cwd(),
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
) -> None:
    """List Sessions isolated to the selected repository."""

    render_session_list(repo, limit=limit)


@session_app.command("show")
def show_session(
    session_id: Annotated[str, typer.Argument()],
    repo: RepoOption = Path.cwd(),
    messages: Annotated[int, typer.Option("--messages", min=0)] = 0,
) -> None:
    """Show Session metadata without exposing transcript content by default."""

    render_session(repo, session_id, messages=messages)


if __name__ == "__main__":
    app()
