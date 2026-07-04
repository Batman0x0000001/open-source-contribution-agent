from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings, create_anthropic_client, load_settings
from osc_agent.tools.pr import draft_pr
from osc_agent.tools.repo import inspect_repo

app = typer.Typer(help="Open Source Contribution Agent Harness")


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
    """执行一次任务后退出，适合脚本化运行和最终验收场景。"""
    settings = load_settings()
    client = create_anthropic_client(settings)
    messages: list[dict[str, object]] = []
    _run_agent_turn(repo=repo, messages=messages, query=task, client=client, settings=settings)
    _print_final_text(messages[-1]["content"])


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


def _print_final_text(content: object) -> None:
    """打印模型最终文本；tool_use block 会被忽略。"""
    if isinstance(content, str):
        typer.echo(content)
        return

    if not isinstance(content, list):
        return

    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                typer.echo(block.get("text", ""))
        elif getattr(block, "type", None) == "text":
            typer.echo(getattr(block, "text", ""))


if __name__ == "__main__":
    app()
