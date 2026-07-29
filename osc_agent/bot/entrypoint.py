"""定义 Bot 常驻进程与运维专用的 osc-agent-bot 入口。"""

from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import ValidationError
import typer

app = typer.Typer(help="Run and maintain the GitHub App Bot services.")


@app.command("control")
def control() -> None:
    """Run the authenticated GitHub webhook Control service."""

    from osc_agent.bot.config import BotSettings
    from osc_agent.bot.control_service import run_control_forever

    try:
        asyncio.run(run_control_forever(BotSettings()))
    except (ValidationError, ValueError, ImportError) as exc:
        _configuration_error(exc)


@app.command("worker")
def worker() -> None:
    """Run the trusted Agent Worker without GitHub App credentials."""

    from osc_agent.bot.config import BotWorkerSettings
    from osc_agent.bot.worker_service import run_worker_forever

    try:
        asyncio.run(run_worker_forever(BotWorkerSettings()))
    except (ValidationError, ValueError, ImportError) as exc:
        _configuration_error(exc)
    except KeyboardInterrupt:
        return


@app.command("doctor")
def doctor(
    control_process: Annotated[
        bool,
        typer.Option("--control", help="Check the credentialed Control service."),
    ] = False,
    worker_process: Annotated[
        bool,
        typer.Option("--worker", help="Check the Docker-enabled Worker service."),
    ] = False,
) -> None:
    """Validate exactly one Bot process boundary."""

    if control_process == worker_process:
        raise typer.BadParameter("select exactly one of --control or --worker")
    from osc_agent.bot.config import BotSettings, BotWorkerSettings
    from osc_agent.bot.doctor import run_bot_control_doctor, run_bot_worker_doctor
    from osc_agent.configuration import load_agent_settings

    try:
        results = (
            asyncio.run(run_bot_control_doctor(BotSettings()))
            if control_process
            else asyncio.run(run_bot_worker_doctor(BotWorkerSettings(), load_agent_settings()))
        )
    except (ValidationError, ValueError, ImportError) as exc:
        _configuration_error(exc)
        return
    _render_doctor(results)


@app.command("cleanup")
def cleanup() -> None:
    """Remove expired terminal Job workspaces and audit records."""

    from osc_agent.bot.cleanup import cleanup_bot_state
    from osc_agent.bot.config import BotMaintenanceSettings

    try:
        workspaces, records = cleanup_bot_state(BotMaintenanceSettings())
    except (ValidationError, ValueError, OSError) as exc:
        _operation_error("cleanup", exc)
        return
    typer.echo(f"PASS\tcleanup\tremoved {workspaces} workspace(s), {records} job record(s)")


@app.command("schema-check")
def schema_check() -> None:
    """Validate the Bot SQLite schema epoch."""

    from osc_agent.bot.config import BotMaintenanceSettings
    from osc_agent.bot.operations import check_schema

    try:
        check_schema(BotMaintenanceSettings())
    except (ValidationError, ValueError, OSError) as exc:
        _operation_error("schema", exc)
        return
    typer.echo("PASS\tschema\tepoch 2 / bot-job-v2")


@app.command("archive-state")
def archive() -> None:
    """Archive Bot SQLite and workspace state for audit."""

    from osc_agent.bot.config import BotMaintenanceSettings
    from osc_agent.bot.operations import archive_state

    try:
        destination = archive_state(BotMaintenanceSettings())
    except (ValidationError, ValueError, OSError) as exc:
        _operation_error("archive", exc)
        return
    typer.echo(str(destination))


@app.command("reset-state")
def reset(
    confirm: Annotated[bool, typer.Option("--confirm", help="Archive then replace Bot state.")] = False,
) -> None:
    """Archive Bot state, then create a fresh epoch-2 database and workspace root."""

    if not confirm:
        raise typer.BadParameter("--confirm is required")
    from osc_agent.bot.config import BotMaintenanceSettings
    from osc_agent.bot.operations import reset_state

    try:
        reset_state(BotMaintenanceSettings())
    except (ValidationError, ValueError, OSError) as exc:
        _operation_error("reset", exc)
        return
    typer.echo("PASS\treset\tepoch 2 state initialized")


@app.command("smoke-test")
def smoke_test() -> None:
    """Check liveness, readiness, and Prometheus without external writes."""

    from osc_agent.bot.config import BotMaintenanceSettings
    from osc_agent.bot.operations import check_service_endpoints

    try:
        endpoints = check_service_endpoints(BotMaintenanceSettings())
    except (ValidationError, ValueError, OSError) as exc:
        _operation_error("smoke", exc)
        return
    for endpoint in endpoints:
        typer.echo(f"PASS\t{endpoint}\tHTTP 200")


@app.command("render-state-machine")
def render_state_machine(
    check: Annotated[bool, typer.Option("--check")] = False,
) -> None:
    """Render or verify the Bot Job state-machine document."""

    from osc_agent.bot.operations import render_state_machine as render_state_machine_document

    try:
        path = render_state_machine_document(check=check)
    except (ValueError, OSError) as exc:
        _operation_error("architecture", exc)
        return
    if not check:
        typer.echo(str(path))


def _render_doctor(results) -> None:
    for item in results:
        typer.echo(f"{item.status}\t{item.name}\t{item.message}")
    if any(item.status == "FAIL" for item in results):
        raise typer.Exit(1)


def _configuration_error(exc: Exception) -> None:
    detail = (
        f"{exc.error_count()} required or invalid settings"
        if isinstance(exc, ValidationError)
        else str(exc)[:500]
    )
    typer.echo(f"FAIL\tconfiguration\t{detail}", err=True)
    raise typer.Exit(1) from exc


def _operation_error(name: str, exc: Exception) -> None:
    typer.echo(f"FAIL\t{name}\t{str(exc)[:500]}", err=True)
    raise typer.Exit(1) from exc
