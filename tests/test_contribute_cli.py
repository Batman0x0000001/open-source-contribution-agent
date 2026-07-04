from __future__ import annotations

from datetime import datetime, timezone
import json

from typer.testing import CliRunner

import osc_agent.cli as cli
from osc_agent.harness.contribution_workflow import design_stage, discover_stage, implement_stage

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


def test_contribute_discover_command_generates_artifacts(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    issues_file = _issues_file(tmp_path)

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


def test_contribute_design_and_draft_pr_commands_read_run(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=_issues_file(tmp_path))

    design_result = runner.invoke(
        cli.app,
        ["contribute", "design", "--repo", str(tmp_path), "--run-id", run.run_id, "--direction", "Tool routing"],
    )
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
