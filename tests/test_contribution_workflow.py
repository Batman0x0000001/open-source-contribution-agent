from __future__ import annotations

from datetime import datetime, timezone
import json

from osc_agent.harness.contribution_workflow import (
    design_stage,
    discover_stage,
    draft_pr_stage,
    implement_stage,
    load_run,
)


def _issues_file(tmp_path):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    path = tmp_path / "issues.json"
    path.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "number": 7,
                        "title": "Add checkpoint docs",
                        "state": "open",
                        "labels": [{"name": "good first issue"}],
                        "updated_at": now,
                        "assignee": None,
                        "assignees": [],
                        "body": "Expected behavior is documented with steps to reproduce the missing workflow.",
                        "html_url": "https://github.com/acme/demo/issues/7",
                    }
                ],
                "comments_by_issue": {"7": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_discover_creates_recoverable_run_and_artifacts(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=_issues_file(tmp_path))

    loaded = load_run(repo_root=tmp_path, run_id=run.run_id)
    artifact = tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "01_discover.json"

    assert loaded.run_id == run.run_id
    assert artifact.exists()
    assert "Add checkpoint docs" in artifact.read_text(encoding="utf-8")


def test_design_implement_and_draft_pr_use_previous_artifacts(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=_issues_file(tmp_path))
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    run = implement_stage(repo_root=tmp_path, run_id=run.run_id, agent_output="done")
    run = draft_pr_stage(repo_root=tmp_path, run_id=run.run_id)
    draft = tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "04_pr_draft.md"

    assert run.stage == "draft_pr"
    assert "**Problem**" in draft.read_text(encoding="utf-8")
    assert "**Notes for Reviewer**" in draft.read_text(encoding="utf-8")


def test_design_requires_discover_artifact(tmp_path):
    missing = tmp_path / ".osc_agent" / "contribution_runs" / "run_missing"
    missing.mkdir(parents=True)
    (missing / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run_missing",
                "repo_root": str(tmp_path),
                "repo_url": "https://github.com/acme/demo",
                "stage": "discover",
                "selected_direction": None,
                "artifacts_dir": str(missing),
            }
        ),
        encoding="utf-8",
    )

    try:
        design_stage(repo_root=tmp_path, run_id="run_missing")
    except ValueError as exc:
        assert "required artifact missing" in str(exc)
    else:
        raise AssertionError("design_stage should require discover artifact")
