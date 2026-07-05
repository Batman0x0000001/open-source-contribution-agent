from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings, create_anthropic_client, load_settings
from osc_agent.harness.contribution_workflow import (
    ContributionRun,
    attach_design_agent_review,
    attach_discover_agent_review,
    design_stage,
    discover_stage,
    draft_pr_stage,
    prepare_implementation_stage,
    record_implementation_result,
)
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
    """启动交互或单次任务模式；子命令由各自命令函数处理。"""
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
    """输出目标仓库的轻量项目地图。"""
    typer.echo(inspect_repo(repo_root=repo))


@app.command("draft-pr")
def draft_pr_command(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
) -> None:
    """基于当前本地 diff 生成 PR 标题和正文草稿，不提交、不推送、不打开 PR。"""
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
    agent_review: Annotated[bool, typer.Option("--agent-review/--no-agent-review")] = False,
) -> None:
    """执行 OpenSourcePR 第 1 步，生成贡献切入点分析。"""
    run = discover_stage(repo_root=repo, repo_url=repo_url, issues_file=issues_file)
    if agent_review:
        review = _run_single_task_capture(repo=repo, task=_artifact_text(run, "01_discover_agent_prompt.md"))
        run = attach_discover_agent_review(repo_root=repo, run_id=run.run_id, review=_content_to_text(review))
    _print_artifact(run, "01_discover.md")


@contribute_app.command("design")
def contribute_design(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
    direction: Annotated[str | None, typer.Option("--direction")] = None,
    agent_review: Annotated[bool, typer.Option("--agent-review/--no-agent-review")] = False,
) -> None:
    """执行 OpenSourcePR 第 2 步，生成技术方案设计。"""
    run = design_stage(repo_root=repo, run_id=run_id, direction=direction)
    if agent_review:
        review = _run_single_task_capture(repo=repo, task=_artifact_text(run, "02_design_agent_prompt.md"))
        run = attach_design_agent_review(repo_root=repo, run_id=run.run_id, review=_content_to_text(review))
    _print_artifact(run, "02_design.md")


@contribute_app.command("implement")
def contribute_implement(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    """执行 OpenSourcePR 第 3 步，先准备 todo/task，再调用现有 agent loop 推进实现。"""
    _confirm_clean_or_continue(repo)
    _, prompt = prepare_implementation_stage(repo_root=repo, run_id=run_id)
    response = _run_single_task_capture(repo=repo, task=prompt)
    run = record_implementation_result(repo_root=repo, run_id=run_id, agent_output=_content_to_text(response))
    _print_artifact(run, "03_implementation_report.md")


@contribute_app.command("draft-pr")
def contribute_draft_pr(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    """执行 OpenSourcePR 第 4 步，生成完整 PR 草稿。"""
    run = draft_pr_stage(repo_root=repo, run_id=run_id)
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
    agent_review: Annotated[bool, typer.Option("--agent-review/--no-agent-review")] = True,
) -> None:
    """按 discover -> design -> implement -> draft-pr 串行执行，阶段之间保留人工确认。"""
    run = discover_stage(repo_root=repo, repo_url=repo_url, issues_file=issues_file)
    if agent_review:
        review = _run_single_task_capture(repo=repo, task=_artifact_text(run, "01_discover_agent_prompt.md"))
        run = attach_discover_agent_review(repo_root=repo, run_id=run.run_id, review=_content_to_text(review))
    _print_artifact(run, "01_discover.md")

    direction = typer.prompt("Choose one contribution direction")
    run = design_stage(repo_root=repo, run_id=run.run_id, direction=direction)
    if agent_review:
        review = _run_single_task_capture(repo=repo, task=_artifact_text(run, "02_design_agent_prompt.md"))
        run = attach_design_agent_review(repo_root=repo, run_id=run.run_id, review=_content_to_text(review))
    _print_artifact(run, "02_design.md")

    if not typer.confirm("Proceed to implementation?", default=False):
        typer.echo(f"Stopped after design. Resume with run id: {run.run_id}")
        return
    _confirm_clean_or_continue(repo)
    _, prompt = prepare_implementation_stage(repo_root=repo, run_id=run.run_id)
    response = _run_single_task_capture(repo=repo, task=prompt)
    run = record_implementation_result(repo_root=repo, run_id=run.run_id, agent_output=_content_to_text(response))
    _print_artifact(run, "03_implementation_report.md")

    run = draft_pr_stage(repo_root=repo, run_id=run.run_id)
    _print_artifact(run, "04_pr_draft.md")


def _run_interactive(repo: Path) -> None:
    """维持多轮消息历史，让用户在同一仓库里持续推进贡献任务。"""
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
    """执行一次 agent loop 并返回最终内容，供 workflow 阶段落盘。"""
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
    """追加用户输入并调用主循环；CLI 只处理 I/O，不实现任何工具细节。"""
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


def _confirm_clean_or_continue(repo: Path) -> None:
    status = git_status(repo_root=repo)
    if status != "(no output)" and not typer.confirm("Working tree has local changes. Continue?", default=False):
        raise typer.Abort()


def _print_artifact(run: ContributionRun, name: str) -> None:
    path = Path(run.artifacts_dir) / name
    typer.echo(path.read_text(encoding="utf-8"))
    typer.echo(f"\n[artifact] {path}")


def _artifact_text(run: ContributionRun, name: str) -> str:
    return (Path(run.artifacts_dir) / name).read_text(encoding="utf-8")


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
    """打印模型最终文本；tool_use block 会被忽略。"""
    text = _content_to_text(content)
    if text:
        typer.echo(text)


if __name__ == "__main__":
    app()
