from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings, create_anthropic_client, load_settings
from osc_agent.harness.contribution_workflow import (
    ContributionRun,
    design_stage,
    discover_stage,
    draft_pr_stage,
    execute_implementation_stage,
)
from osc_agent.harness.gates import GateResult, gate_design, gate_discover, gate_implementation
from osc_agent.harness.worktree import create_worktree, worktree_path
from osc_agent.tools.git import git_status
from osc_agent.tools.pr import draft_pr
from osc_agent.tools.repo import inspect_repo

app = typer.Typer(help="Open Source Contribution Agent Harness")
contribute_app = typer.Typer(help="Run the OpenSourcePR 1-4 contribution workflow.")
app.add_typer(contribute_app, name="contribute")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    repo: Annotated[
        Path | None,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ] = None,
    task: Annotated[str | None, typer.Option("--task", help="Run one contribution task and exit.")] = None,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if repo is None:
        raise typer.BadParameter("--repo is required unless a subcommand provides it.")
    if task:
        _run_single_task(repo=repo, task=task)
        return
    _run_interactive(repo)


@app.command("inspect")
def inspect_command(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
) -> None:
    typer.echo(inspect_repo(repo_root=repo))


@app.command("draft-pr")
def draft_pr_command(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
) -> None:
    typer.echo(draft_pr(repo_root=repo))


@contribute_app.command("discover")
def contribute_discover(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    repo_url: Annotated[str, typer.Option("--repo-url")],
    issues_file: Annotated[
        Path | None,
        typer.Option("--issues-file", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    ] = None,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Use LLM analysis (requires ANTHROPIC_API_KEY); use --no-llm for local fallback.",
        ),
    ] = True,
) -> None:
    client, settings = _stage_client() if use_llm else (None, None)
    run = discover_stage(repo_root=repo, repo_url=repo_url, issues_file=issues_file, client=client, settings=settings)
    _print_artifact(run, "01_discover.md")


@contribute_app.command("design")
def contribute_design(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
    direction: Annotated[str | None, typer.Option("--direction")] = None,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Use LLM analysis (requires ANTHROPIC_API_KEY); use --no-llm for local fallback.",
        ),
    ] = True,
) -> None:
    client, settings = _stage_client() if use_llm else (None, None)
    run = design_stage(repo_root=repo, run_id=run_id, direction=direction, client=client, settings=settings)
    _print_artifact(run, "02_design.md")


@contribute_app.command("implement")
def contribute_implement(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    _confirm_clean_or_continue(repo)
    work_repo = _create_run_worktree(repo=repo, run_id=run_id)
    run = _execute_implementation(work_repo, run_id)
    _print_gate(gate_implementation(Path(run.artifacts_dir), work_repo))
    _print_artifact(run, "03_implementation_report.md")


@contribute_app.command("draft-pr")
def contribute_draft_pr(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Use LLM analysis (requires ANTHROPIC_API_KEY); use --no-llm for local fallback.",
        ),
    ] = True,
) -> None:
    client, settings = _stage_client() if use_llm else (None, None)
    run = draft_pr_stage(repo_root=repo, run_id=run_id, client=client, settings=settings)
    _print_artifact(run, "04_pr_draft.md")


@contribute_app.command("run")
def contribute_run(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    repo_url: Annotated[str, typer.Option("--repo-url")],
    issues_file: Annotated[
        Path | None,
        typer.Option("--issues-file", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    ] = None,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Use LLM analysis (requires ANTHROPIC_API_KEY); use --no-llm for local fallback.",
        ),
    ] = True,
) -> None:
    client, settings = _stage_client() if use_llm else (None, None)

    run = discover_stage(repo_root=repo, repo_url=repo_url, issues_file=issues_file, client=client, settings=settings)
    _require_gate(gate_discover(Path(run.artifacts_dir)))
    _print_artifact(run, "01_discover.md")

    direction = typer.prompt("Choose one contribution direction", default=_default_direction_label(run))
    run = design_stage(repo_root=repo, run_id=run.run_id, direction=direction, client=client, settings=settings)
    _require_gate(gate_design(Path(run.artifacts_dir)))
    _print_artifact(run, "02_design.md")

    if not typer.confirm("Proceed to implementation?", default=False):
        typer.echo(f"Stopped after design. Resume with run id: {run.run_id}")
        return

    _confirm_clean_or_continue(repo)
    work_repo = _create_run_worktree(repo=repo, run_id=run.run_id)
    run = _execute_implementation(work_repo, run.run_id)
    _require_gate(gate_implementation(Path(run.artifacts_dir), work_repo))
    _print_artifact(run, "03_implementation_report.md")

    run = draft_pr_stage(repo_root=work_repo, run_id=run.run_id, client=client, settings=settings)
    _print_artifact(run, "04_pr_draft.md")


def _run_interactive(repo: Path) -> None:
    settings = load_settings()
    client = create_anthropic_client(settings)
    messages: list[dict[str, object]] = []

    typer.echo("osc-agent: open source contribution harness")
    typer.echo("Type q, quit, or exit to stop.\n")

    while True:
        try:
            query = typer.prompt("osc-agent")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break

        if query.strip().lower() in {"q", "quit", "exit", ""}:
            break

        _run_agent_turn(repo=repo, messages=messages, query=query, client=client, settings=settings)
        _print_final_text(messages[-1]["content"])


def _run_single_task(*, repo: Path, task: str) -> None:
    _print_final_text(_run_single_task_capture(repo=repo, task=task))


def _run_single_task_capture(*, repo: Path, task: str) -> object:
    settings = load_settings()
    client = create_anthropic_client(settings)
    messages: list[dict[str, object]] = []
    _run_agent_turn(repo=repo, messages=messages, query=task, client=client, settings=settings)
    return messages[-1]["content"]


def _run_agent_turn(
    *,
    repo: Path,
    messages: list[dict[str, object]],
    query: str,
    client: Any,
    settings: Settings,
) -> None:
    messages.append({"role": "user", "content": query})
    log_dir = repo / ".osc_agent"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / "agent.log").open("a", encoding="utf-8") as log:
        agent_loop(
            messages,
            client=client,
            settings=settings,
            repo_root=repo,
            output=log,
            confirm=lambda prompt: typer.confirm(prompt, default=False),
        )


def _stage_client() -> tuple[Any | None, Settings | None]:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise typer.BadParameter(
            "ANTHROPIC_API_KEY is required for --llm. Set it in .env or use --no-llm "
            "for the deterministic local fallback."
        )
    try:
        return create_anthropic_client(settings), settings
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _execute_implementation(repo: Path, run_id: str) -> ContributionRun:
    def run_step(stage: str, prompt: str) -> str:
        typer.echo(f"[implementation:{stage}]")
        return _content_to_text(_run_single_task_capture(repo=repo, task=prompt))

    try:
        return execute_implementation_stage(repo_root=repo, run_id=run_id, run_step=run_step)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _confirm_clean_or_continue(repo: Path) -> None:
    status = git_status(repo_root=repo)
    if status != "(no output)" and not typer.confirm("Working tree has local changes. Continue?", default=False):
        raise typer.Abort()


def _create_run_worktree(*, repo: Path, run_id: str) -> Path:
    name = f"contribution-{run_id}"[:64]
    result = create_worktree(repo_root=repo, name=name, task_id="")
    if result.startswith("Error:") or result.startswith("Git error:"):
        raise typer.BadParameter(f"Could not create implementation worktree: {result}")
    path = worktree_path(repo, name)
    _copy_run_artifacts(repo, path, run_id)
    typer.echo(f"[worktree] {path}")
    return path


def _copy_run_artifacts(source_repo: Path, work_repo: Path, run_id: str) -> None:
    source = source_repo / ".osc_agent" / "contribution_runs" / run_id
    target = work_repo / ".osc_agent" / "contribution_runs" / run_id
    if not source.exists():
        raise typer.BadParameter(f"Contribution run artifacts not found: {run_id}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    run_path = target / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["repo_root"] = str(work_repo.resolve())
    payload["artifacts_dir"] = str(target.resolve())
    run_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_gate(result: GateResult) -> None:
    _print_gate(result)
    if not result.passed:
        raise typer.Abort()


def _print_gate(result: GateResult) -> None:
    status = "passed" if result.passed else "failed"
    typer.echo(f"[gate:{status}] {result.reason}")
    for warning in result.warnings:
        typer.echo(f"[gate:warning] {warning}")


def _print_artifact(run: ContributionRun, name: str) -> None:
    path = Path(run.artifacts_dir) / name
    typer.echo(path.read_text(encoding="utf-8"))
    typer.echo(f"\n[artifact] {path}")


def _default_direction_label(run: ContributionRun) -> str:
    path = Path(run.artifacts_dir) / "01_discover.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    directions = payload.get("top_directions") or []
    if not directions:
        return ""
    return str(directions[0].get("name", ""))


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part)


def _print_final_text(content: object) -> None:
    text = _content_to_text(content)
    if text:
        typer.echo(text)


if __name__ == "__main__":
    app()
