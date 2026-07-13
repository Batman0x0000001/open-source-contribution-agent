from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from typer.testing import CliRunner

import osc_agent.cli as cli
from osc_agent.workflows.contribution import (
    design_stage,
    discover_stage,
    implement_stage,
    load_run,
    update_design_contract,
)

runner = CliRunner()


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


def test_contribute_discover_command_generates_artifacts(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    issues_file = _issues_file(tmp_path)
    _init_repo(tmp_path)

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
            "--no-llm",
        ],
    )

    assert result.exit_code == 0
    assert "Top 3" in result.output
    assert "Improve tool routing" in result.output


def test_contribute_design_and_draft_pr_commands_read_run(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
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
            "Tool routing",
            "--no-llm",
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
        ["contribute", "draft-pr", "--repo", str(tmp_path), "--run-id", run.run_id, "--no-llm"],
    )

    assert design_result.exit_code == 0
    assert "技术方案设计" in design_result.output
    assert draft_result.exit_code == 0
    assert "**Problem**" in draft_result.output
    assert "**Notes for Reviewer**" in draft_result.output


def test_contribute_update_design_updates_structured_contract(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Tool routing")

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


def test_contribute_resume_advances_completed_discovery(tmp_path):
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    _init_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)

    result = runner.invoke(
        cli.app,
        ["contribute", "resume", "--repo", str(tmp_path), "--run-id", run.run_id, "--no-llm"],
    )

    assert result.exit_code == 0, result.output
    resumed = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert resumed.stage == "design"
    assert resumed.stage_status["design"] == "SUCCEEDED"


def test_llm_mode_requires_anthropic_api_key(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: cli.Settings(None, None, "test-model", None))

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
    assert "ANTHROPIC_API_KEY is required for --llm" in result.output
    assert "--no-llm" in result.output


def test_llm_option_help_documents_api_key_requirement():
    result = runner.invoke(cli.app, ["contribute", "discover", "--help"])

    assert result.exit_code == 0
    assert "Use LLM analysis" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "--no-llm" in result.output


def test_worktree_uses_source_run_as_authoritative_state(tmp_path):
    source_repo = tmp_path / "source"
    work_repo = tmp_path / "worktree"
    source_repo.mkdir()
    work_repo.mkdir()
    _init_repo(source_repo)
    run = discover_stage(repo_root=source_repo, repo_url="https://github.com/acme/demo")

    cli._copy_run_artifacts(source_repo, work_repo, run.run_id)

    copied = load_run(repo_root=work_repo, run_id=run.run_id)
    expected_artifacts = source_repo / ".osc_agent" / "contribution_runs" / run.run_id
    assert Path(copied.repo_root) == source_repo.resolve()
    assert Path(copied.artifacts_dir) == expected_artifacts.resolve()
    assert Path(copied.worktree_root) == work_repo.resolve()
    assert not (work_repo / ".osc_agent" / "contribution_runs").exists()
