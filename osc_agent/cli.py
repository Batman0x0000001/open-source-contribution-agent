"""定义 osc-agent 命令行入口及其子命令。"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from pydantic import JsonValue

from osc_agent.application import (
    AgentApplicationConfig,
    AgentProfile,
    SkillInput,
    UserPrompt,
    build_agent_application,
    build_session_store,
    build_skill_catalog,
)
from osc_agent.cli_session import render_session_summary, run_conversation
from osc_agent.config import load_settings
from osc_agent.doctor import run_doctor
from osc_agent.application.session_summary import build_session_summary
from osc_agent.runtime.messages import RuntimeMessage, TextBlock, ToolResultBlock, ToolUseBlock
from osc_agent.runtime.tool_models import ApprovalResponse, Ask


app = typer.Typer(help="Extensible coding agent for open-source contribution workflows.")
skill_app = typer.Typer(help="List and run validated Skills.")
session_app = typer.Typer(help="Inspect repository-scoped Sessions.")
bot_app = typer.Typer(help="Run the optional GitHub App control and worker services.")
deploy_app = typer.Typer(help="Install and operate the Bot deployment.")
architecture_app = typer.Typer(help="Render architecture contracts.")
app.add_typer(skill_app, name="skill")
app.add_typer(session_app, name="session")
app.add_typer(bot_app, name="bot")
app.add_typer(deploy_app, name="deploy")
app.add_typer(architecture_app, name="architecture")

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
    settings = load_settings()
    agent_application = build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=repo,
            profile=AgentProfile(
                profile_id="local_debug",
                system_prompt="Use repository evidence and the smallest safe change that satisfies the task.",
            ),
            approval_handler=_approve,
            question_handler=_ask_questions,
        )
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
        session_id = build_session_store(repo).latest_session_id()
        if session_id is None:
            raise typer.BadParameter("no valid Session exists for this repository")
    assert session_id is not None
    settings = load_settings()
    agent_application = build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=repo,
            profile=AgentProfile(
                profile_id="local_debug",
                system_prompt="Continue the existing repository task from its authoritative transcript.",
            ),
            approval_handler=_approve,
            question_handler=_ask_questions,
        )
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
    settings = load_settings()
    session_id = str(uuid4())
    agent_application = build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=repo,
            profile=AgentProfile(
                profile_id="local_debug",
                system_prompt="Use repository evidence and follow the explicitly invoked Skill.",
                allowed_initial_skills=frozenset({name}),
            ),
            approval_handler=_approve,
            question_handler=_ask_questions,
        )
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


async def _approve(decision: Ask) -> ApprovalResponse:
    typer.echo(f"\nTool: {decision.tool_name}")
    typer.echo(f"Working directory: {decision.working_directory}")
    typer.echo(f"Risk: {decision.risk}")
    typer.echo("Input preview:")
    typer.echo(json.dumps(decision.preview, ensure_ascii=False, indent=2))
    if decision.tool_name == "bash":
        typer.echo(
            "WARNING: this Bash command runs on the Host without an OS sandbox and may access files available to the current user."
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


@bot_app.command("serve")
def bot_serve() -> None:
    """Run the authenticated GitHub webhook control service."""

    try:
        from osc_agent.bot.config import BotSettings
        from osc_agent.bot.service_runner import run_control_forever

        settings = BotSettings()
    except (ImportError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    asyncio.run(run_control_forever(settings))


@bot_app.command("worker")
def bot_worker() -> None:
    """Run the trusted Agent worker without loading GitHub App credentials."""

    from osc_agent.bot.config import BotWorkerSettings
    from osc_agent.bot.service_runner import run_worker_forever

    try:
        asyncio.run(run_worker_forever(BotWorkerSettings()))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except KeyboardInterrupt:
        return


@bot_app.command("doctor")
def bot_doctor(
    control: Annotated[
        bool,
        typer.Option("--control", help="Check the credentialed Control service without Docker access."),
    ] = False,
    worker: Annotated[
        bool,
        typer.Option("--worker", help="Check the Docker-enabled Worker service."),
    ] = False,
) -> None:
    """Validate exactly one Bot process boundary."""

    from pydantic import ValidationError
    from osc_agent.bot.config import BotSettings, BotWorkerSettings
    from osc_agent.bot.doctor import run_bot_control_doctor, run_bot_worker_doctor

    if control == worker:
        raise typer.BadParameter("select exactly one of --control or --worker")
    try:
        results = (
            asyncio.run(run_bot_control_doctor(BotSettings()))
            if control
            else asyncio.run(run_bot_worker_doctor(BotWorkerSettings(), load_settings()))
        )
    except (ValidationError, ValueError) as exc:
        detail = (
            f"{exc.error_count()} required or invalid settings"
            if isinstance(exc, ValidationError)
            else str(exc)[:500]
        )
        typer.echo(f"FAIL\tconfiguration\t{detail}", err=True)
        raise typer.Exit(1) from exc
    for item in results:
        typer.echo(f"{item.status}\t{item.name}\t{item.message}")
    if any(item.status == "FAIL" for item in results):
        raise typer.Exit(1)


@bot_app.command("cleanup")
def bot_cleanup() -> None:
    """Remove expired terminal Job workspaces and audit records."""

    from pydantic import ValidationError
    from osc_agent.bot.cleanup import cleanup_bot_state
    from osc_agent.bot.config import BotSettings

    try:
        workspaces, records = cleanup_bot_state(BotSettings())
    except (ValidationError, ValueError, OSError) as exc:
        typer.echo(f"FAIL\tcleanup\t{str(exc)[:500]}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"PASS\tcleanup\tremoved {workspaces} workspace(s), {records} job record(s)")


@deploy_app.command("schema-check")
def deploy_schema_check() -> None:
    from osc_agent.bot.store import BotStore

    value = os.environ.get("OSC_AGENT_BOT_DATABASE_PATH")
    if not value:
        raise typer.BadParameter("OSC_AGENT_BOT_DATABASE_PATH is required")
    BotStore(Path(value)).check_schema()
    typer.echo("PASS\tschema\tepoch 2 / bot-job-v2")


@deploy_app.command("archive-state")
def deploy_archive_state() -> None:
    database = Path(os.environ["OSC_AGENT_BOT_DATABASE_PATH"]).resolve()
    workspace = Path(os.environ["OSC_AGENT_BOT_WORKSPACE_ROOT"]).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = database.parent / "archive" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for candidate in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
        if candidate.exists():
            shutil.copy2(candidate, destination / candidate.name)
    if workspace.exists():
        shutil.copytree(workspace, destination / "workspaces")
    typer.echo(str(destination))


@deploy_app.command("reset-state")
def deploy_reset_state(
    confirm: Annotated[bool, typer.Option("--confirm", help="Archive then replace Bot state.")] = False,
) -> None:
    """Archive Bot state, then create a fresh epoch-2 database and workspace root."""

    if not confirm:
        raise typer.BadParameter("--confirm is required")
    deploy_archive_state()
    database = Path(os.environ["OSC_AGENT_BOT_DATABASE_PATH"]).resolve()
    workspace = Path(os.environ["OSC_AGENT_BOT_WORKSPACE_ROOT"]).resolve()
    for candidate in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
        if candidate.exists():
            candidate.unlink()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o2770)
    from osc_agent.bot.store import BotStore
    BotStore(database).initialize()
    typer.echo("PASS\treset\tepoch 2 state initialized")


@deploy_app.command("doctor")
def deploy_doctor() -> None:
    """Run the production Control and Worker preflight checks."""

    from osc_agent.bot.config import BotSettings, BotWorkerSettings
    from osc_agent.bot.doctor import run_bot_control_doctor, run_bot_worker_doctor
    results = asyncio.run(run_bot_control_doctor(BotSettings()))
    results += asyncio.run(run_bot_worker_doctor(BotWorkerSettings(), load_settings()))
    for item in results:
        typer.echo(f"{item.status}\t{item.name}\t{item.message}")
    if any(item.status == "FAIL" for item in results):
        raise typer.Exit(1)


@deploy_app.command("smoke-test")
def deploy_smoke_test() -> None:
    """Check liveness, readiness, and Prometheus without external write side effects."""

    from osc_agent.bot.config import BotSettings
    settings = BotSettings()
    base = f"http://127.0.0.1:{settings.bind_port}"
    for endpoint in ("/health/live", "/health/ready", "/metrics"):
        try:
            with urllib.request.urlopen(base + endpoint, timeout=10) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}")
        except Exception as exc:
            typer.echo(f"FAIL\t{endpoint}\t{str(exc)[:300]}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"PASS\t{endpoint}\tHTTP 200")


@architecture_app.command("render-state-machine")
def render_state_machine(
    check: Annotated[bool, typer.Option("--check")] = False,
) -> None:
    from osc_agent.bot.state_machine import BotJobStateMachine

    path = Path(__file__).resolve().parents[1] / "docs" / "architecture" / "bot-job-state-machine.md"
    content = (
        "# Bot Job State Machine\n\n"
        "This file is generated by `osc-agent architecture render-state-machine`.\n\n"
        "```mermaid\n" + BotJobStateMachine.mermaid() + "\n```\n\n"
        + BotJobStateMachine.transition_table() + "\n"
    )
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise typer.Exit(1)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    typer.echo(str(path))


if __name__ == "__main__":
    app()
