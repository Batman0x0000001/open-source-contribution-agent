"""验证诊断检查的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os
from pathlib import Path
import subprocess

from osc_agent.config import Settings
from osc_agent.doctor import run_doctor
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.messages import RuntimeMessage, TextBlock


def _git_repo(root: Path) -> None:
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)


def test_local_doctor_checks_required_dependencies_without_exposing_secrets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    secret = "secret-do-not-print"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CUSTOM_BUILD_FLAG", "visible-to-tools")
    monkeypatch.setenv("GITHUB_TOKEN", secret)

    results = asyncio.run(
        run_doctor(
            repository_root=repo,
            settings=Settings(
                anthropic_api_key=secret,
                model_id="test-model",
                subprocess_env_allowlist=frozenset(
                    {"CUSTOM_BUILD_FLAG", "GITHUB_TOKEN"}
                ),
            ),
            local_only=True,
        )
    )

    failures = {item.name for item in results if item.status == "FAIL"}
    assert failures == ({"bash"} if os.name == "nt" else set())
    assert next(item for item in results if item.name == "model_connectivity").status == "WARN"
    assert next(item for item in results if item.name == "execution_isolation").status == "WARN"
    subprocess_check = next(
        item for item in results if item.name == "subprocess_environment"
    )
    assert subprocess_check.status == "PASS"
    assert "GITHUB_TOKEN" in subprocess_check.message
    assert secret not in "\n".join(item.message for item in results)


def test_live_doctor_uses_gateway_without_creating_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    state = tmp_path / "state"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(state))

    class Gateway:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            yield ModelCompleted(
                message=RuntimeMessage(
                    role="assistant",
                    content=[TextBlock(text="OK")],
                ),
                stop_reason="end_turn",
            )

    monkeypatch.setattr("osc_agent.doctor.build_model_gateway", lambda _settings: Gateway())

    results = asyncio.run(
        run_doctor(
            repository_root=repo,
            settings=Settings(
                anthropic_api_key="secret",
                model_id="test-model",
            ),
            local_only=False,
        )
    )

    assert next(item for item in results if item.name == "model_connectivity").status == "PASS"
    assert not list(state.rglob("*.jsonl"))


def test_doctor_marks_missing_model_configuration_as_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    results = asyncio.run(
        run_doctor(
            repository_root=repo,
            settings=Settings(),
            local_only=True,
        )
    )

    failed = {item.name for item in results if item.status == "FAIL"}
    assert {"MODEL_ID", "ANTHROPIC_API_KEY", "agents"} <= failed


def test_doctor_reports_invalid_external_skill_as_warning(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    invalid = repo / ".osc_agent" / "skills" / "legacy" / "SKILL.md"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(
        """---
name: legacy
description: Legacy
when_to_use: Never
output_schema: {type: object}
---
Legacy.
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(tmp_path / "state"))

    results = asyncio.run(
        run_doctor(
            repository_root=repo,
            settings=Settings(model_id="test-model"),
            local_only=True,
        )
    )

    skills = next(item for item in results if item.name == "skills")
    assert skills.status == "WARN"
    assert "INVALID_SKILL_MANIFEST" in skills.message
