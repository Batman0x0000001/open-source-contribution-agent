"""
启动程序
  ↓
解析 --repo 参数
  ↓
加载配置 settings
  ↓
创建 Anthropic client
  ↓
初始化 messages 对话历史
  ↓
进入 while 循环，等待用户输入
  ↓
把用户输入加入 messages
  ↓
调用 agent_loop(...)
  ↓
打印模型最终回复
  ↓
继续等待下一轮输入
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from osc_agent.agent_loop import agent_loop
from osc_agent.config import create_anthropic_client, load_settings

app = typer.Typer(help="Open Source Contribution Agent Harness")


@app.callback(invoke_without_command=True)
def main(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
) -> None:
    """启动交互式 CLI，把用户输入交给最小 agent loop。"""
    settings = load_settings()
    client = create_anthropic_client(settings)
    messages: list[dict[str, object]] = []

    typer.echo("osc-agent s01: minimal agent loop + bash")
    typer.echo("Type q, quit, or exit to stop.\n")

    while True:
        try:
            query = typer.prompt("osc-agent")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break

        if query.strip().lower() in {"q", "quit", "exit", ""}:
            break

        messages.append({"role": "user", "content": query})
        agent_loop(messages, client=client, settings=settings, repo_root=repo, output=None)
        _print_final_text(messages[-1]["content"])


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
