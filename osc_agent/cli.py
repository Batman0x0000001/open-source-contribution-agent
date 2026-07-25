from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from pydantic import JsonValue

from osc_agent.application import ApplicationServices, build_application, build_skill_catalog
from osc_agent.config import load_settings
from osc_agent.runtime.models import Ask, AssistantDelta, CapabilityScope, ResumeQueryParams, RunStopped, RuntimeEvent, RuntimeMessage, StartQueryParams, TextBlock
from osc_agent.skills.models import SkillInvocation


app = typer.Typer(help="Claude Code style extensible coding agent.")
skill_app = typer.Typer(help="List and run validated Skills.")
app.add_typer(skill_app, name="skill")

RepoOption = Annotated[Path, typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True)]


@app.command("run")
def run_agent(
    repo: RepoOption,
    prompt: Annotated[str | None, typer.Argument(help="Task for the general coding agent.")] = None,
) -> None:
    """Start a new general AgentRuntime Session."""

    task = prompt or typer.prompt("Task")
    session_id = str(uuid4())
    services = _application(repo)
    settings = load_settings()

    async def execute() -> None:
        events = services.runtime.query(
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
        )
        await _render(events)

    typer.echo(f"Session: {session_id}")
    asyncio.run(execute())


@app.command("resume")
def resume_session(
    session_id: Annotated[str, typer.Argument(help="Existing Session ID.")],
    repo: RepoOption,
    prompt: Annotated[str | None, typer.Option("--prompt", help="Optional new user message.")] = None,
) -> None:
    """Resume a Session transcript without contribution-stage knowledge."""

    services = _application(repo)
    messages = [RuntimeMessage(role="user", content=[TextBlock(text=prompt)])] if prompt else []

    async def execute() -> None:
        await _render(
            services.runtime.query(
                ResumeQueryParams(
                    session_id=session_id,
                    messages=messages,
                    repository_root=str(repo),
                    config=services.query_config,
                )
            )
        )

    asyncio.run(execute())


@app.command("contribute")
def contribute(
    repo: RepoOption,
    repo_url: Annotated[str, typer.Option("--repo-url")],
    goal: Annotated[str | None, typer.Option("--goal")] = None,
) -> None:
    """Invoke the single open-source-contribution Skill."""

    _run_inline_skill("open-source-contribution", repo, {"repo_url": repo_url, "goal": goal})


@skill_app.command("list")
def list_skills(repo: RepoOption = Path.cwd()) -> None:
    """List frontmatter metadata without loading Skill bodies or resources."""

    for descriptor in build_skill_catalog(repo).list():
        typer.echo(f"{descriptor.manifest.name}\t{descriptor.source}\t{descriptor.manifest.description}")


@skill_app.command("run")
def run_skill(
    name: Annotated[str, typer.Argument(help="Exact Skill name.")],
    repo: RepoOption,
    arguments: Annotated[str, typer.Option("--arguments", help="Strict JSON object.")] = "{}",
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
        _run_inline_skill(name, repo, payload)
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


def _run_inline_skill(name: str, repo: Path, arguments: dict[str, JsonValue]) -> None:
    services = _application(repo)
    session_id = str(uuid4())
    typer.echo(f"Session: {session_id}")

    async def execute() -> None:
        await _render(
            services.skill_command_runner.run(
                name=name,
                arguments=arguments,
                session_id=session_id,
                working_directory=str(repo),
            )
        )

    try:
        asyncio.run(execute())
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


async def _approve(decision: Ask) -> bool:
    typer.echo(f"\nTool: {decision.tool_name}")
    typer.echo(f"Working directory: {decision.working_directory}")
    typer.echo(f"Risk: {decision.risk}")
    typer.echo("Input preview:")
    typer.echo(json.dumps(decision.preview, ensure_ascii=False, indent=2))
    return typer.confirm(decision.prompt, default=False)


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


async def _render(events) -> None:
    stopped = False
    async for event in events:
        event = event  # type: RuntimeEvent
        if isinstance(event, AssistantDelta):
            typer.echo(event.text, nl=False)
        elif isinstance(event, RunStopped):
            stopped = True
            typer.echo(f"\nStopped: {event.transition.reason}", err=True)
    typer.echo()
    if stopped:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
