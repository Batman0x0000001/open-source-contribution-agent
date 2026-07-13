from __future__ import annotations

import subprocess

from osc_agent.agent_loop import build_tool_handlers
from osc_agent.tools.pr import build_pr_draft, format_pr_draft
from osc_agent.workflows.contribution import (
    bind_run_worktree,
    build_workflow_pr_draft,
    design_stage,
    discover_stage,
    record_implementation_result,
    update_design_contract,
)


def _issues_file(tmp_path):
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    path = tmp_path / "issues.json"
    path.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "number": 9,
                        "title": "Improve eval docs",
                        "state": "open",
                        "labels": [{"name": "good first issue"}],
                        "updated_at": now,
                        "assignee": None,
                        "assignees": [],
                        "body": "Expected behavior is documented with steps to reproduce missing guidance.",
                    }
                ],
                "comments_by_issue": {"9": []},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_pr_draft_uses_structured_sections_for_docs_change():
    diff = "diff --git a/README.md b/README.md\n"
    status = " M README.md"

    draft = build_pr_draft(diff=diff, status=status)
    output = format_pr_draft(draft)

    assert draft.title == "Update documentation"
    assert "## Summary" in output
    assert "## Tests" in output
    assert "## Risk" in output
    assert "README.md" in output
    assert "documentation-only" in output


def test_agent_handlers_expose_draft_pr(tmp_path):
    handlers = build_tool_handlers(tmp_path)

    assert "draft_pr" in handlers


def test_workflow_pr_draft_extracts_testing_from_report(tmp_path):
    from osc_agent.tools.pr import draft_pr

    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text("def run_agent():\n    return 'ok'\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Improve eval docs")
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
    record_implementation_result(repo_root=tmp_path, run_id=run.run_id, agent_output="pytest tests\n2 passed")

    output = draft_pr(repo_root=tmp_path, run_id=run.run_id)

    assert "**Testing**" in output
    assert "2 passed" in output
    assert "**Notes for Reviewer**" in output

    worktree = tmp_path / "implementation-worktree"
    worktree.mkdir()
    bind_run_worktree(repo_root=tmp_path, run_id=run.run_id, worktree_root=worktree)
    worktree_output = build_workflow_pr_draft(repo_root=worktree, run_id=run.run_id)
    assert "**Problem**" in worktree_output
    assert "2 passed" in worktree_output
