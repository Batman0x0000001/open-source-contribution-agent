"""验证运行状态路径的契约、边界条件与回归行为。"""

from __future__ import annotations

from pathlib import Path

from osc_agent.runtime.state_paths import ApplicationStatePaths, default_state_root


def test_repository_state_is_deterministic_and_isolated(tmp_path: Path) -> None:
    state_root = tmp_path / "user-state"
    first_repo = tmp_path / "repo-one"
    second_repo = tmp_path / "repo-two"
    first_repo.mkdir()
    second_repo.mkdir()

    first = ApplicationStatePaths.for_repository(first_repo, state_root=state_root)
    repeated = ApplicationStatePaths.for_repository(first_repo, state_root=state_root)
    second = ApplicationStatePaths.for_repository(second_repo, state_root=state_root)

    assert first.repository == repeated.repository
    assert first.repository != second.repository
    assert first.repository.is_relative_to(state_root.resolve())
    assert not first.repository.is_relative_to(first_repo.resolve())


def test_state_root_environment_override(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "override"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(override))
    assert default_state_root() == override
