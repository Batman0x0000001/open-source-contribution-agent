"""验证Runtime YAML 配置的契约、边界条件与回归行为。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from osc_agent.configuration import AgentSettings, load_agent_settings
from osc_agent.bot.config import load_repository_catalog
from osc_agent.configuration.runtime import default_runtime_config_path, load_runtime_config


def _runtime_yaml(*, verify_rounds: int = 16, extra: str = "") -> str:
    return f"""\
agents:
  main:
    max_rounds: 30
    max_total_tokens: 200000
    deadline_seconds: 1800
    max_output_tokens: 8192
    max_no_progress_rounds: 6
  explore:
    max_rounds: 8
    max_total_tokens: 40000
    deadline_seconds: 300
    max_output_tokens: 4096
    max_no_progress_rounds: 3
  verify:
    max_rounds: {verify_rounds}
    max_total_tokens: 60000
    deadline_seconds: 900
    max_output_tokens: 8192
    max_no_progress_rounds: 3
model_retry:
  max_attempts: 3
  base_seconds: 1
  max_seconds: 8
{extra}"""


def test_packaged_runtime_config_contains_all_execution_budgets() -> None:
    config = load_runtime_config(default_runtime_config_path())

    assert config.agents.main.to_query_config().max_rounds == 30
    assert config.agents.explore.to_query_config().max_total_tokens == 40_000
    assert config.agents.verify.to_query_config().max_rounds == 16
    assert config.model_retry.to_retry_policy().max_attempts == 3


def test_ubuntu_production_config_combines_runtime_and_repositories() -> None:
    path = Path(__file__).parents[2] / "deploy" / "ubuntu" / "config.example.yml"

    runtime = load_runtime_config(path)
    catalog = load_repository_catalog(path)

    assert runtime.agents.main.max_rounds == 30
    assert "owner/repository" in catalog.repositories


def test_settings_loads_runtime_config_selected_by_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.yml"
    path.write_text(_runtime_yaml(verify_rounds=21), encoding="utf-8")
    monkeypatch.setenv("OSC_AGENT_RUNTIME_CONFIG", str(path))

    settings = load_agent_settings()

    assert settings.runtime_config_path == path
    assert settings.runtime.agents.verify.max_rounds == 21


def test_agent_settings_construction_does_not_read_runtime_file(tmp_path: Path) -> None:
    runtime = load_runtime_config(default_runtime_config_path())

    settings = AgentSettings(
        model_id="test-model",
        runtime_config_path=tmp_path / "missing.yml",
        runtime=runtime,
    )

    assert settings.runtime is runtime


@pytest.mark.parametrize(
    "content, error",
    [
        (_runtime_yaml(verify_rounds=0), "greater than or equal to 1"),
        (_runtime_yaml(extra="unknown: true\n"), "Extra inputs are not permitted"),
        (_runtime_yaml().replace("  max_seconds: 8", "  max_seconds: 0"), "must not exceed"),
        ("agents: [", "unable to read runtime config"),
    ],
)
def test_runtime_config_rejects_invalid_files(
    tmp_path: Path,
    content: str,
    error: str,
) -> None:
    path = tmp_path / "runtime.yml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises((ValueError, ValidationError), match=error):
        load_runtime_config(path)
