from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings, create_anthropic_client, load_settings
from osc_agent.harness.capabilities import AgentCapabilityScope
from osc_agent.harness.worktree import create_worktree, worktree_path
from osc_agent.tools.git import git_status
from osc_agent.tools.repo import inspect_repo
from osc_agent.workflows.contribution import (
    ContributionRun,
    GateResult,
    bind_run_worktree,
    configure_run,
    design_stage,
    discover_stage,
    draft_pr_stage,
    execute_implementation_stage,
    load_run,
    record_test_waiver,
    update_design_contract,
    gate_design,
    gate_discover,
    gate_implementation,
)

app = typer.Typer(help="Open Source Contribution Agent Harness")
contribute_app = typer.Typer(help="Run the OpenSourcePR 1-4 contribution workflow.")

# 根命令只保留通用能力；有状态的贡献流程统一放在 contribute 命令组中。
app.add_typer(contribute_app, name="contribute")


# 根命令入口
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


# 贡献阶段按 discover -> design -> implement -> draft-pr 的执行顺序排列。
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
) -> None:
    client, settings = _stage_client()

    # 检查 GITHUB_TOKEN 配置，避免未授权 API 访问受限
    if issues_file is None and not settings.github_token:
        typer.echo("⚠️  Warning: GITHUB_TOKEN not configured", err=True)
        typer.echo("   Without token: 60 requests/hour | With token: 5,000 requests/hour", err=True)
        typer.echo("   Set GITHUB_TOKEN in .env or use --issues-file for offline mode", err=True)
        typer.echo("", err=True)

    run = discover_stage(
        repo_root=repo,
        repo_url=repo_url,
        issues_file=issues_file,
        client=client,
        settings=settings,
    )
    _print_artifact(run, "01_discover.md")


@contribute_app.command("design")
def contribute_design(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
    direction: Annotated[str | None, typer.Option("--direction")] = None,
) -> None:
    client, settings = _stage_client()
    try:
        run = design_stage(
            repo_root=repo,
            run_id=run_id,
            direction=direction,
            client=client,
            settings=settings,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_artifact(run, "02_design.md")


@contribute_app.command("update-design")
def contribute_update_design(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
    allow_file: Annotated[list[str] | None, typer.Option("--allow-file")] = None,
    allow_new_dir: Annotated[list[str] | None, typer.Option("--allow-new-dir")] = None,
    target_symbol: Annotated[list[str] | None, typer.Option("--target-symbol")] = None,
    test_command: Annotated[list[str] | None, typer.Option("--test-command")] = None,
    requirement_id: Annotated[list[str] | None, typer.Option("--requirement-id")] = None,
    task_type: Annotated[str | None, typer.Option("--task-type")] = None,
    baseline_command: Annotated[str | None, typer.Option("--baseline-command")] = None,
    baseline_output: Annotated[str | None, typer.Option("--baseline-output")] = None,
    baseline_exit_code: Annotated[int, typer.Option("--baseline-exit-code")] = 1,
    reproduction_test_file: Annotated[list[str] | None, typer.Option("--reproduction-test-file")] = None,
    max_files: Annotated[int | None, typer.Option("--max-files", min=1)] = None,
    max_diff_lines: Annotated[int | None, typer.Option("--max-diff-lines", min=1)] = None,
) -> None:
    run = load_run(repo_root=repo, run_id=run_id)
    design_path = Path(run.artifacts_dir) / "02_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    updates: dict[str, Any] = {}
    if allow_file is not None:
        updates["allowed_files"] = allow_file
        updates["files_to_modify"] = allow_file
        updates["impact_area"] = allow_file
    if allow_new_dir is not None:
        updates["allowed_new_dirs"] = allow_new_dir
    if target_symbol is not None:
        updates["target_symbols"] = target_symbol
    if test_command is not None:
        updates["tests_to_run"] = test_command
        updates["acceptance_checks"] = [
            {
                "criterion": f"Command succeeds: {command}",
                "command": command,
                "manual_check": False,
                "requirement_ids": requirement_id or [],
            }
            for command in test_command
        ] or design.get("acceptance_checks", [])
    elif requirement_id is not None:
        raise typer.BadParameter("--requirement-id requires --test-command")
    if task_type is not None:
        normalized_task_type = task_type.strip().lower()
        if normalized_task_type not in {"behavior", "docs", "config"}:
            raise typer.BadParameter("--task-type must be behavior, docs, or config")
        updates["task_type"] = normalized_task_type
    if baseline_command is not None or baseline_output is not None:
        if not baseline_command or not baseline_output:
            raise typer.BadParameter("--baseline-command and --baseline-output must be provided together")
        updates["baseline_checks"] = [
            {
                "command": baseline_command,
                "expected_exit_codes": [baseline_exit_code],
                "output_contains": baseline_output,
            }
        ]
        updates["reproduction_mode"] = "existing"
        updates["reproduction_test_files"] = []
    if reproduction_test_file is not None:
        if baseline_command is not None or baseline_output is not None:
            raise typer.BadParameter(
                "choose either an existing baseline or generated reproduction test files, not both"
            )
        updates["reproduction_mode"] = "generated_test"
        updates["reproduction_test_files"] = reproduction_test_file
        updates["baseline_checks"] = []
    if max_files is not None:
        updates["max_changed_files"] = max_files
    if max_diff_lines is not None:
        updates["max_diff_lines"] = max_diff_lines
    if not updates:
        raise typer.BadParameter("provide at least one design contract option to update")
    run = update_design_contract(repo_root=repo, run_id=run_id, updates=updates)
    _print_artifact(run, "02_design.md")


@contribute_app.command("implement")
def contribute_implement(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
    max_rounds: Annotated[int | None, typer.Option("--max-rounds", min=1)] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens", min=1)] = None,
    deadline_seconds: Annotated[int | None, typer.Option("--deadline-seconds", min=1)] = None,
    max_files: Annotated[int | None, typer.Option("--max-files", min=1)] = None,
    max_diff_lines: Annotated[int | None, typer.Option("--max-diff-lines", min=1)] = None,
    test_waiver_reason: Annotated[str | None, typer.Option("--test-waiver-reason")] = None,
) -> None:
    settings = _settings_with_overrides(
        load_settings(), max_rounds, max_tokens, deadline_seconds, max_files, max_diff_lines
    )
    run = load_run(repo_root=repo, run_id=run_id)
    implement_status = (run.stage_status or {}).get("implement")
    if implement_status == "PENDING":
        run = configure_run(repo_root=repo, run_id=run_id, settings=settings)
    elif max_files is not None or max_diff_lines is not None:
        raise typer.BadParameter(
            "implementation scope budgets are frozen after implementation starts"
        )
    work_repo = _implementation_repo(repo=repo, run=run)
    run = _execute_implementation(work_repo, run_id, settings=settings)
    if test_waiver_reason:
        run = record_test_waiver(repo_root=work_repo, run_id=run_id, reason=test_waiver_reason)
    _require_gate(gate_implementation(Path(run.artifacts_dir), work_repo))
    _print_artifact(run, "03_implementation_report.md")


@contribute_app.command("draft-pr")
def contribute_draft_pr(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    client, settings = _stage_client()
    run = load_run(repo_root=repo, run_id=run_id)
    active_repo = Path(run.worktree_root or repo)
    if not active_repo.is_dir():
        raise typer.BadParameter(f"bound implementation worktree does not exist: {active_repo}")
    run = draft_pr_stage(repo_root=active_repo, run_id=run_id, client=client, settings=settings)
    _print_artifact(run, "04_pr_draft.md")


# 工作流编排入口：resume 恢复已有 Run，run 串联一次完整的新 Run。
@contribute_app.command("resume")
def contribute_resume(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    ],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    """从权威 Run 状态和最近一个持久化 checkpoint 继续工作流。"""
    run = load_run(repo_root=repo, run_id=run_id)
    status = (run.stage_status or {}).get(run.stage)
    client, settings = _stage_client()

    if run.stage == "discover":
        if status != "SUCCEEDED":
            raise typer.BadParameter("Discovery did not complete; start a new run to refresh repository evidence.")
        run = design_stage(
            repo_root=repo,
            run_id=run_id,
            direction=run.selected_direction or _default_direction_label(run),
            client=client,
            settings=settings,
        )
        _print_artifact(run, "02_design.md")
        return

    if run.stage == "design":
        if status != "SUCCEEDED":
            run = design_stage(
                repo_root=repo,
                run_id=run_id,
                direction=run.selected_direction or _default_direction_label(run),
                client=client,
                settings=settings,
            )
        configure_run(repo_root=repo, run_id=run_id, settings=settings)
        work_repo = _implementation_repo(repo=repo, run=run)
        run = _execute_implementation(work_repo, run_id, settings=settings)
        _print_artifact(run, "03_implementation_report.md")
        return

    if run.stage == "implement":
        if status != "SUCCEEDED":
            work_repo = _implementation_repo(repo=repo, run=run)
            run = _execute_implementation(work_repo, run_id, settings=settings)
            _print_artifact(run, "03_implementation_report.md")
            return
        active_repo = Path(run.worktree_root or repo)
        run = draft_pr_stage(repo_root=active_repo, run_id=run_id, client=client, settings=settings)
        _print_artifact(run, "04_pr_draft.md")
        return

    if run.stage == "draft_pr" and status != "SUCCEEDED":
        active_repo = Path(run.worktree_root or repo)
        run = draft_pr_stage(repo_root=active_repo, run_id=run_id, client=client, settings=settings)
        _print_artifact(run, "04_pr_draft.md")
        return
    typer.echo(f"Run {run_id} is already complete.")


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
    max_rounds: Annotated[int | None, typer.Option("--max-rounds", min=1)] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens", min=1)] = None,
    deadline_seconds: Annotated[int | None, typer.Option("--deadline-seconds", min=1)] = None,
    max_files: Annotated[int | None, typer.Option("--max-files", min=1)] = None,
    max_diff_lines: Annotated[int | None, typer.Option("--max-diff-lines", min=1)] = None,
    test_waiver_reason: Annotated[str | None, typer.Option("--test-waiver-reason")] = None,
) -> None:
    client, loaded_settings = _stage_client()
    settings = _settings_with_overrides(
        loaded_settings, max_rounds, max_tokens, deadline_seconds, max_files, max_diff_lines
    )

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

    _require_clean_repository(repo)
    work_repo = _create_run_worktree(repo=repo, run_id=run.run_id)
    run = _execute_implementation(work_repo, run.run_id, settings=settings)
    if test_waiver_reason:
        run = record_test_waiver(repo_root=work_repo, run_id=run.run_id, reason=test_waiver_reason)
    _require_gate(gate_implementation(Path(run.artifacts_dir), work_repo))
    _print_artifact(run, "03_implementation_report.md")

    run = draft_pr_stage(repo_root=work_repo, run_id=run.run_id, client=client, settings=settings)
    _print_artifact(run, "04_pr_draft.md")


# 通用 Agent 运行辅助函数
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


def _run_single_task_capture(
    *,
    repo: Path,
    task: str,
    settings: Settings | None = None,
    client: Any | None = None,
    capabilities: AgentCapabilityScope | None = None,
    objective: str | None = None,
) -> object:
    settings = settings or load_settings()
    client = client or create_anthropic_client(settings)
    messages: list[dict[str, object]] = []
    result = _run_agent_turn(
        repo=repo,
        messages=messages,
        query=task,
        client=client,
        settings=settings,
        capabilities=capabilities,
        objective=objective,
    )
    if result.status.value != "SUCCESS":
        raise RuntimeError(f"{result.status.value}: {result.reason}")
    return messages[-1]["content"]


def _run_agent_turn(
    *,
    repo: Path,
    messages: list[dict[str, object]],
    query: str,
    client: Any,
    settings: Settings,
    capabilities: AgentCapabilityScope | None = None,
    objective: str | None = None,
) -> Any:
    messages.append({"role": "user", "content": query})
    log_dir = repo / ".osc_agent"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / "agent.log").open("a", encoding="utf-8") as log:
        return agent_loop(
            messages,
            client=client,
            settings=settings,
            repo_root=repo,
            output=log,
            confirm=lambda prompt: typer.confirm(prompt, default=False),
            capabilities=capabilities,
            objective=objective,
        )


# 贡献工作流辅助函数
def _stage_client() -> tuple[Any, Settings]:
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise typer.BadParameter(
            "ANTHROPIC_API_KEY is required for the contribution workflow. Set it in .env."
        )
    try:
        return create_anthropic_client(settings), settings
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _execute_implementation(repo: Path, run_id: str, *, settings: Settings | None = None) -> ContributionRun:
    active_settings = settings or load_settings()
    client: Any | None = None

    def run_step(stage: str, prompt: str) -> str:
        nonlocal client
        if client is None:
            client = create_anthropic_client(active_settings)
        typer.echo(f"[implementation:{stage}]")
        run = load_run(repo_root=repo, run_id=run_id)
        design_path = Path(run.artifacts_dir) / "02_design.json"
        design = json.loads(design_path.read_text(encoding="utf-8"))
        return _content_to_text(
            _run_single_task_capture(
                repo=repo,
                task=prompt,
                settings=active_settings,
                client=client,
                capabilities=AgentCapabilityScope.contribution(stage, design, run_id=run_id),
                objective=run.selected_direction,
            )
        )

    try:
        return execute_implementation_stage(
            repo_root=repo,
            run_id=run_id,
            run_step=run_step,
            github_token=active_settings.github_token,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _require_clean_repository(repo: Path) -> None:
    status = git_status(repo_root=repo)
    external = [line for line in status.splitlines() if ".osc_agent/" not in line.replace("\\", "/")]
    if status != "(no output)" and any(line.strip() for line in external):
        raise typer.BadParameter("Working tree must be clean before creating an implementation worktree.")


def _create_run_worktree(*, repo: Path, run_id: str) -> Path:
    run = load_run(repo_root=repo, run_id=run_id)
    from osc_agent.tools.git import git_head

    if git_head(repo_root=repo).strip() != run.base_commit_sha:
        raise typer.BadParameter("STALE_RUN: repository HEAD differs from the saved base commit.")
    name = f"contribution-{run_id}"[:64]
    result = create_worktree(repo_root=repo, name=name, task_id="")
    if result.startswith("Error:") or result.startswith("Git error:"):
        raise typer.BadParameter(f"Could not create implementation worktree: {result}")
    path = worktree_path(repo, name)
    bind_run_worktree(repo_root=repo, run_id=run_id, worktree_root=path)
    typer.echo(f"[worktree] {path}")
    return path


def _copy_run_artifacts(source_repo: Path, work_repo: Path, run_id: str) -> None:
    """兼容旧调用名：现在只创建指向权威 Run 的引用，不再复制状态。"""
    bind_run_worktree(repo_root=source_repo, run_id=run_id, worktree_root=work_repo)


def _implementation_repo(*, repo: Path, run: ContributionRun) -> Path:
    if run.worktree_root:
        worktree = Path(run.worktree_root)
        if worktree.is_dir():
            return worktree
    _require_clean_repository(repo)
    return _create_run_worktree(repo=repo, run_id=run.run_id)


def _settings_with_overrides(
    settings: Settings,
    max_rounds: int | None,
    max_tokens: int | None,
    deadline_seconds: int | None,
    max_files: int | None,
    max_diff_lines: int | None,
) -> Settings:
    updates = {
        name: value
        for name, value in {
            "max_agent_rounds": max_rounds,
            "max_total_tokens": max_tokens,
            "agent_deadline_seconds": deadline_seconds,
            "max_changed_files": max_files,
            "max_diff_lines": max_diff_lines,
        }.items()
        if value is not None
    }
    return settings.model_copy(update=updates)


# Gate 与输出辅助函数
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
