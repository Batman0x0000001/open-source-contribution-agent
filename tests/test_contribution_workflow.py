from __future__ import annotations

from datetime import datetime, timezone
import json

from osc_agent.harness.contribution_workflow import (
    attach_design_agent_review,
    attach_discover_agent_review,
    build_design_review_prompt,
    design_stage,
    discover_stage,
    draft_pr_stage,
    implement_stage,
    load_run,
    prepare_implementation_stage,
    record_implementation_result,
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
    assert (artifact.parent / "01_discover_agent_prompt.md").exists()


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


def test_agent_reviews_are_attached_to_artifacts(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=_issues_file(tmp_path))

    attach_discover_agent_review(repo_root=tmp_path, run_id=run.run_id, review="Deep source analysis")
    design = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    attach_design_agent_review(repo_root=tmp_path, run_id=run.run_id, review="Concrete implementation plan")

    discover_md = (tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "01_discover.md").read_text(encoding="utf-8")
    design_md = (tmp_path / ".osc_agent" / "contribution_runs" / design.run_id / "02_design.md").read_text(encoding="utf-8")

    assert "Deep source analysis" in discover_md
    assert "Concrete implementation plan" in design_md


def test_prepare_implementation_happens_before_recording_result(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=_issues_file(tmp_path))
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")

    prepared, prompt = prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    report_path = tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "03_implementation_report.md"

    assert prepared.stage == "implement"
    assert "Implementation has not run yet" in report_path.read_text(encoding="utf-8")
    assert "Before editing" in prompt

    record_implementation_result(
        repo_root=tmp_path,
        run_id=run.run_id,
        agent_output="pytest tests/test_demo.py\n1 passed",
    )

    report = report_path.read_text(encoding="utf-8")
    assert "1 passed" in report
    assert "Implementation has not run yet" not in report


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


def test_design_review_prompt_focuses_selected_issue_only():
    discover = {
        "repo_url": "https://github.com/acme/demo",
        "top_directions": [
            {"name": "Issue #7: Add checkpoint docs", "entry": "docs/checkpoint.md"},
            {"name": "Issue #9: Add unrelated metrics", "entry": "metrics.py"},
        ],
        "candidate_issues": [
            {"number": 7, "title": "Add checkpoint docs", "body": "selected"},
            {"number": 9, "title": "Add unrelated metrics", "body": "not selected"},
        ],
        "issue_scores": [
            {"number": 7, "score": 90},
            {"number": 9, "score": 80},
        ],
        "architecture_dimensions": [{"dimension": "docs", "location": "docs/checkpoint.md"}],
        "evidence_pack": {"symbols": {"tool": [{"file": "tool.py"} for _ in range(20)]}},
        "agent_review": "x" * 5000,
    }

    prompt = build_design_review_prompt(discover=discover, selected="Issue #7: Add checkpoint docs")

    assert "Issue #7: Add checkpoint docs" in prompt
    assert "Add checkpoint docs" in prompt
    assert "Add unrelated metrics" not in prompt
    assert len(prompt) < 14_000
    build_design_review_prompt,
