from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import osc_agent.cli as cli
from osc_agent.workflows.contribution import (
    design_stage as _design_stage,
    discover_stage as _discover_stage,
    implement_stage,
    load_run,
    update_design_contract,
)

runner = CliRunner()
from stage_llm_fakes import FakeDiscoverClient, fake_settings, run_fake_discover


def discover_stage(**kwargs):
    return run_fake_discover(_discover_stage, **kwargs)


def design_stage(**kwargs):
    kwargs.setdefault("client", FakeDiscoverClient())
    kwargs.setdefault("settings", fake_settings())
    return _design_stage(**kwargs)


def _issues_file(tmp_path):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    path = tmp_path / "issues.json"
    path.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "number": 5,
                        "title": "Improve tool routing",
                        "state": "open",
                        "labels": [{"name": "enhancement"}],
                        "updated_at": now,
                        "assignee": None,
                        "assignees": [],
                        "body": "Expected behavior includes clear tool routing steps to reproduce the issue.",
                        "html_url": "https://github.com/acme/demo/issues/5",
                    }
                ],
                "comments_by_issue": {"5": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def _init_repo(path):
    (path / "agent.py").write_text("def run_agent():\n    return 'ok'\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_contribute_discover_command_generates_artifacts(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    issues_file = _issues_file(tmp_path)
    _init_repo(tmp_path)
    monkeypatch.setattr(cli, "_stage_client", lambda: (FakeDiscoverClient(), fake_settings()))

    result = runner.invoke(
        cli.app,
        [
            "contribute",
            "discover",
            "--repo",
            str(tmp_path),
            "--repo-url",
            "https://github.com/acme/demo",
            "--issues-file",
            str(issues_file),
        ],
    )

    assert result.exit_code == 0
    assert "Top 3" in result.output
    assert "Improve tool routing" in result.output


def test_settings_overrides_copy_pydantic_settings():
    settings = fake_settings()

    updated = cli._settings_with_overrides(
        settings,
        max_rounds=12,
        max_tokens=None,
        deadline_seconds=90,
        max_files=2,
        max_diff_lines=None,
    )

    assert updated is not settings
    assert updated.anthropic_api_key == settings.anthropic_api_key
    assert updated.model_id == settings.model_id
    assert updated.max_agent_rounds == 12
    assert updated.max_total_tokens == settings.max_total_tokens
    assert updated.agent_deadline_seconds == 90
    assert updated.max_changed_files == 2
    assert updated.max_diff_lines == settings.max_diff_lines


def test_contribute_design_reports_contract_error_without_traceback(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.setattr(cli, "_stage_client", lambda: (object(), fake_settings()))
    monkeypatch.setattr(
        cli,
        "design_stage",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError(
                "allowed_files may contain only existing repository files: "
                "tests/test_new.py"
            )
        ),
    )

    result = runner.invoke(
        cli.app,
        [
            "contribute",
            "design",
            "--repo",
            str(tmp_path),
            "--run-id",
            "run_test",
            "--direction",
            "issue:1",
        ],
    )

    assert result.exit_code != 0
    assert "allowed_files may contain only existing repository files" in result.output
    assert "Traceback" not in result.output


def test_contribute_design_and_draft_pr_commands_read_run(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
    monkeypatch.setattr(cli, "_stage_client", lambda: (FakeDiscoverClient(), fake_settings()))
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)

    design_result = runner.invoke(
        cli.app,
        [
            "contribute",
            "design",
            "--repo",
            str(tmp_path),
            "--run-id",
            run.run_id,
            "--direction",
            "Issue #5: Improve tool routing",
        ],
    )
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )
    (tmp_path / "README.md").write_text("# Demo\n\nUpdated.\n", encoding="utf-8")
    implement_stage(repo_root=tmp_path, run_id=run.run_id, agent_output="done")
    draft_result = runner.invoke(
        cli.app,
        ["contribute", "draft-pr", "--repo", str(tmp_path), "--run-id", run.run_id],
    )

    assert design_result.exit_code == 0
    assert "技术方案设计" in design_result.output
    assert draft_result.exit_code == 0
    assert "**Problem**" in draft_result.output
    assert "**Notes for Reviewer**" in draft_result.output


def test_contribute_update_design_updates_structured_contract(tmp_path):
    (tmp_path / "README.md").write_text("# Tool routing\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #5: Improve tool routing")

    result = runner.invoke(
        cli.app,
        [
            "contribute",
            "update-design",
            "--repo",
            str(tmp_path),
            "--run-id",
            run.run_id,
            "--allow-file",
            "README.md",
            "--target-symbol",
            "Tool routing",
            "--test-command",
            'python -c "print(1)"',
            "--max-files",
            "1",
            "--max-diff-lines",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    design = json.loads((Path(run.artifacts_dir) / "02_design.json").read_text(encoding="utf-8"))
    assert design["allowed_files"] == ["README.md"]
    assert design["target_symbols"] == ["Tool routing"]
    assert design["max_changed_files"] == 1
    assert design["max_diff_lines"] == 20
    assert design["source_evidence"][0]["file"] == "README.md"


def test_contribute_update_design_records_behavior_failure_baseline(tmp_path):
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #5: Improve tool routing")

    result = runner.invoke(
        cli.app,
        [
            "contribute", "update-design",
            "--repo", str(tmp_path),
            "--run-id", run.run_id,
            "--allow-file", "agent.py",
            "--target-symbol", "run_agent",
            "--test-command", "python -m pytest tests/test_agent.py",
            "--task-type", "behavior",
            "--baseline-command", "python -m pytest tests/test_agent.py",
            "--baseline-output", "AssertionError",
        ],
    )

    assert result.exit_code == 0, result.output
    design = json.loads((Path(run.artifacts_dir) / "02_design.json").read_text(encoding="utf-8"))
    spec = design["contribution_spec"]
    assert spec["task_type"] == "behavior"
    assert spec["baseline_checks"] == [
        {
            "command": "python -m pytest tests/test_agent.py",
            "expected_exit_codes": [1],
            "output_contains": "AssertionError",
        }
    ]
    assert spec["reproduction"]["mode"] == "existing"
    assert design["source_evidence"][0]["symbol"] == "run_agent"


def test_contribute_update_design_declares_generated_regression_test(tmp_path):
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #5: Improve tool routing")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_agent.py"],
            "baseline_checks": [{
                "command": "python -m pytest -q tests/test_agent.py",
                "expected_exit_codes": [1],
                "output_contains": "AssertionError",
            }],
            "reproduction_mode": "existing",
        },
    )

    result = runner.invoke(
        cli.app,
        [
            "contribute", "update-design",
            "--repo", str(tmp_path),
            "--run-id", run.run_id,
            "--allow-file", "agent.py",
            "--allow-new-dir", "tests",
            "--target-symbol", "run_agent",
            "--test-command", "python -m pytest -q tests/test_agent.py",
            "--requirement-id", "REQ-1",
            "--task-type", "behavior",
            "--reproduction-test-file", "tests/test_agent.py",
        ],
    )

    assert result.exit_code == 0, result.output
    design = json.loads((Path(run.artifacts_dir) / "02_design.json").read_text(encoding="utf-8"))
    assert design["contribution_spec"]["reproduction"] == {
        "mode": "generated_test",
        "command": "python -m pytest -q tests/test_agent.py",
        "test_files": ["tests/test_agent.py"],
    }
    assert design["contribution_spec"]["baseline_checks"] == []
    assert design["acceptance_checks"][0]["requirement_ids"] == ["REQ-1"]


def test_contribute_resume_advances_completed_discovery(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    monkeypatch.setattr(cli, "_stage_client", lambda: (FakeDiscoverClient(), fake_settings()))

    result = runner.invoke(
        cli.app,
        ["contribute", "resume", "--repo", str(tmp_path), "--run-id", run.run_id],
    )

    assert result.exit_code == 0, result.output
    resumed = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert resumed.stage == "design"
    assert resumed.stage_status["design"] == "SUCCEEDED"


def test_llm_mode_requires_anthropic_api_key(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: cli.Settings(model_id="test-model"))

    result = runner.invoke(
        cli.app,
        [
            "contribute",
            "discover",
            "--repo",
            str(tmp_path),
            "--repo-url",
            "https://github.com/acme/demo",
        ],
    )

    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY is required for the contribution workflow" in result.output
    assert "--no-llm" not in result.output


@pytest.mark.parametrize("command", ["discover", "design", "draft-pr", "resume", "run"])
def test_contribution_help_does_not_offer_non_llm_mode(command):
    result = runner.invoke(cli.app, ["contribute", command, "--help"])

    assert result.exit_code == 0
    assert "--llm" not in result.output
    assert "--no-llm" not in result.output


def test_worktree_uses_source_run_as_authoritative_state(tmp_path):
    source_repo = tmp_path / "source"
    work_repo = tmp_path / "worktree"
    source_repo.mkdir()
    _init_repo(source_repo)
    run = discover_stage(repo_root=source_repo, repo_url="https://github.com/acme/demo")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(work_repo), "HEAD"],
        cwd=source_repo,
        check=True,
        capture_output=True,
    )

    cli._copy_run_artifacts(source_repo, work_repo, run.run_id)

    copied = load_run(repo_root=work_repo, run_id=run.run_id)
    expected_artifacts = source_repo / ".osc_agent" / "contribution_runs" / run.run_id
    assert Path(copied.repo_root) == source_repo.resolve()
    assert Path(copied.artifacts_dir) == expected_artifacts.resolve()
    assert Path(copied.worktree_root) == work_repo.resolve()
    assert not (work_repo / ".osc_agent" / "contribution_runs").exists()


def test_implementation_repo_reuses_bound_worktree(tmp_path, monkeypatch):
    worktree = tmp_path / "existing-worktree"
    worktree.mkdir()
    monkeypatch.setattr(
        cli,
        "_create_run_worktree",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not create another worktree")),
    )

    selected = cli._implementation_repo(
        repo=tmp_path,
        run=SimpleNamespace(run_id="run_test", worktree_root=str(worktree)),
    )

    assert selected == worktree


def test_contribute_implement_retry_does_not_reconfigure_frozen_design(tmp_path, monkeypatch):
    run = SimpleNamespace(
        run_id="run_test",
        worktree_root=str(tmp_path),
        artifacts_dir=str(tmp_path),
        stage_status={"implement": "FAILED"},
    )
    monkeypatch.setattr(cli, "load_settings", fake_settings)
    monkeypatch.setattr(cli, "load_run", lambda **_kwargs: run)
    monkeypatch.setattr(
        cli,
        "configure_run",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not mutate frozen design")),
    )
    monkeypatch.setattr(cli, "_execute_implementation", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(cli, "gate_implementation", lambda *_args, **_kwargs: cli.GateResult(True, "ok"))
    monkeypatch.setattr(cli, "_print_artifact", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        cli.app,
        ["contribute", "implement", "--repo", str(tmp_path), "--run-id", run.run_id],
    )

    assert result.exit_code == 0, result.output


def test_contribute_draft_pr_uses_bound_implementation_worktree(tmp_path, monkeypatch):
    source_repo = tmp_path / "source"
    worktree = source_repo / ".osc_agent" / "worktrees" / "contribution-run_test"
    worktree.mkdir(parents=True)
    run = SimpleNamespace(
        run_id="run_test",
        worktree_root=str(worktree),
        artifacts_dir=str(source_repo),
    )
    called_with = []
    monkeypatch.setattr(cli, "_stage_client", lambda: (object(), fake_settings()))
    monkeypatch.setattr(cli, "load_run", lambda **_kwargs: run)
    monkeypatch.setattr(
        cli,
        "draft_pr_stage",
        lambda **kwargs: called_with.append(kwargs["repo_root"]) or run,
    )
    monkeypatch.setattr(cli, "_print_artifact", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        cli.app,
        ["contribute", "draft-pr", "--repo", str(source_repo), "--run-id", run.run_id],
    )

    assert result.exit_code == 0, result.output
    assert called_with == [worktree]
