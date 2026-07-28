"""执行本地 Runtime、模型和外部工具的诊断检查。"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from osc_agent.agents.explore import build_explore_registration
from osc_agent.agents.registry import AgentRegistry
from osc_agent.agents.verify import build_verify_registration
from osc_agent.composition import build_model_gateway, build_skill_catalog
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelCompleted, ModelRequest
from osc_agent.runtime.models import FrozenContractModel, RuntimeMessage, TextBlock
from osc_agent.runtime.session_store import FileSessionStore
from osc_agent.runtime.state_paths import ApplicationStatePaths
from osc_agent.tools.process_runner import (
    build_subprocess_environment,
    is_protected_environment_name,
)


class DiagnosticResult(FrozenContractModel):
    name: str = Field(min_length=1)
    status: Literal["PASS", "WARN", "FAIL"]
    message: str = Field(min_length=1, max_length=500)


async def run_doctor(
    *,
    repository_root: Path,
    settings: Settings,
    local_only: bool,
) -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    subprocess_environment = build_subprocess_environment(
        settings.subprocess_env_allowlist
    )
    results.append(
        DiagnosticResult(
            name="python",
            status="PASS" if sys.version_info >= (3, 10) else "FAIL",
            message=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    for package in ("anthropic", "pydantic", "typer", "portalocker"):
        try:
            version = importlib.metadata.version(package)
            results.append(DiagnosticResult(name=package, status="PASS", message=version))
        except importlib.metadata.PackageNotFoundError:
            results.append(
                DiagnosticResult(name=package, status="FAIL", message="package is not installed")
            )

    results.append(
        DiagnosticResult(
            name="MODEL_ID",
            status="PASS" if settings.model_id else "FAIL",
            message="configured" if settings.model_id else "missing",
        )
    )
    results.append(
        DiagnosticResult(
            name="ANTHROPIC_API_KEY",
            status="PASS" if settings.anthropic_api_key else "FAIL",
            message="configured" if settings.anthropic_api_key else "missing",
        )
    )
    results.append(_executable_check("bash", ["--version"], subprocess_environment))
    results.append(_executable_check("rg", ["--version"], subprocess_environment))
    results.append(_executable_check("git", ["--version"], subprocess_environment))
    results.append(_git_repository_check(repository_root, subprocess_environment))
    results.append(_state_directory_check(repository_root))
    protected_names = sorted(
        name for name in os.environ if is_protected_environment_name(name)
    )
    results.append(
        DiagnosticResult(
            name="subprocess_environment",
            status="PASS",
            message=(
                f"{len(subprocess_environment)} variable(s) allowed; protected names excluded: "
                + (", ".join(protected_names) if protected_names else "none detected")
            )[:500],
        )
    )
    results.append(
        DiagnosticResult(
            name="execution_isolation",
            status="WARN",
            message="Host execution with filtered subprocess environment; no OS sandbox",
        )
    )

    try:
        catalog = build_skill_catalog(repository_root)
        names = [item.manifest.name for item in catalog.list()]
        results.append(
            DiagnosticResult(
                name="skills",
                status="WARN" if catalog.blocked_overrides() else "PASS",
                message=(
                    f"{len(names)} Skill(s) discovered"
                    + (
                        f"; {len(catalog.blocked_overrides())} reserved-name override(s) blocked"
                        if catalog.blocked_overrides()
                        else ""
                    )
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 - Doctor 必须聚合所有诊断。
        results.append(
            DiagnosticResult(name="skills", status="FAIL", message=_safe_error(exc, settings))
        )

    if settings.model_id:
        try:
            registry = AgentRegistry(
                [
                    build_explore_registration(
                        model=settings.model_id,
                        config=settings.runtime.agents.explore.to_query_config(),
                    ),
                    build_verify_registration(
                        model=settings.model_id,
                        config=settings.runtime.agents.verify.to_query_config(),
                    ),
                ]
            )
            results.append(
                DiagnosticResult(
                    name="agents",
                    status="PASS",
                    message=f"{len(registry.list())} built-in Agent(s) registered",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                DiagnosticResult(name="agents", status="FAIL", message=_safe_error(exc, settings))
            )
    else:
        results.append(
            DiagnosticResult(
                name="agents",
                status="FAIL",
                message="MODEL_ID is required to register built-in Agents",
            )
        )

    results.append(
        DiagnosticResult(
            name="GITHUB_TOKEN",
            status="PASS" if os.environ.get("GITHUB_TOKEN") else "WARN",
            message="configured" if os.environ.get("GITHUB_TOKEN") else "not configured; public API limits apply",
        )
    )

    if local_only:
        results.append(
            DiagnosticResult(
                name="model_connectivity",
                status="WARN",
                message="skipped by --local-only",
            )
        )
    elif not settings.model_id or not settings.anthropic_api_key:
        results.append(
            DiagnosticResult(
                name="model_connectivity",
                status="FAIL",
                message="model configuration is incomplete",
            )
        )
    else:
        try:
            gateway = build_model_gateway(settings)
            completed = False
            async for event in gateway.stream(
                ModelRequest(
                    model=settings.model_id,
                    system_prompt="Return exactly OK.",
                    messages=[
                        RuntimeMessage(
                            role="user",
                            content=[TextBlock(text="Connectivity check.")],
                        )
                    ],
                    tools=[],
                    max_output_tokens=8,
                )
            ):
                completed = completed or isinstance(event, ModelCompleted)
            if not completed:
                raise ValueError("model returned no completed response")
            results.append(
                DiagnosticResult(
                    name="model_connectivity",
                    status="PASS",
                    message="minimal model request completed",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                DiagnosticResult(
                    name="model_connectivity",
                    status="FAIL",
                    message=_safe_error(exc, settings),
                )
            )
    return results


def _executable_check(
    name: str,
    arguments: list[str],
    environment: dict[str, str],
) -> DiagnosticResult:
    executable = shutil.which(name, path=environment.get("PATH"))
    if executable is None:
        return DiagnosticResult(name=name, status="FAIL", message="executable not found")
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DiagnosticResult(name=name, status="FAIL", message=type(exc).__name__)
    output = (result.stdout or result.stderr).splitlines()
    return DiagnosticResult(
        name=name,
        status="PASS" if result.returncode == 0 else "FAIL",
        message=(output[0][:300] if output else f"exit code {result.returncode}"),
    )


def _git_repository_check(
    repository_root: Path,
    environment: dict[str, str],
) -> DiagnosticResult:
    executable = shutil.which("git", path=environment.get("PATH"))
    if executable is None:
        return DiagnosticResult(name="git_repository", status="FAIL", message="git not found")
    result = subprocess.run(
        [
            executable,
            "-c",
            f"safe.directory={repository_root.resolve()}",
            "-C",
            str(repository_root),
            "rev-parse",
            "--show-toplevel",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
        check=False,
        env=environment,
    )
    return DiagnosticResult(
        name="git_repository",
        status="PASS" if result.returncode == 0 else "FAIL",
        message=(
            "repository detected"
            if result.returncode == 0
            else "selected path is not a Git repository"
        ),
    )


def _state_directory_check(repository_root: Path) -> DiagnosticResult:
    paths = ApplicationStatePaths.for_repository(repository_root)
    probe = paths.repository / f".doctor-{uuid4().hex}.tmp"
    lock_path = paths.sessions / ".locks" / "doctor-probe.lock"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        with FileSessionStore(paths.sessions).lease("doctor-probe"):
            pass
        if lock_path.exists():
            lock_path.unlink()
        return DiagnosticResult(
            name="state_directory",
            status="PASS",
            message="write and Session lock probe succeeded",
        )
    except (OSError, ValueError) as exc:
        return DiagnosticResult(
            name="state_directory",
            status="FAIL",
            message=type(exc).__name__,
        )
    finally:
        for target in (probe, lock_path):
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass


def _safe_error(exc: Exception, settings: Settings) -> str:
    message = str(exc) or type(exc).__name__
    for secret in (settings.anthropic_api_key, os.environ.get("GITHUB_TOKEN")):
        if secret:
            message = message.replace(secret, "<redacted>")
    return message[:500]
