from __future__ import annotations

from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path

from osc_agent.workflows.contribution import (
    attach_design_agent_review,
    attach_discover_agent_review,
    build_design_review_prompt,
    design_stage,
    discover_stage,
    draft_pr_stage,
    execute_implementation_stage,
    implement_stage,
    load_run,
    prepare_implementation_stage,
    record_implementation_result,
    record_test_waiver,
    run_verification_commands,
    validate_design_files,
    validate_implementation_scope,
    update_design_contract,
)
from osc_agent.workflows.contribution.workflow import _write_raw_json
from osc_agent.harness.contracts import RunStatus


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


def _prepare_repo(tmp_path):
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text("def run_agent():\n    return 'ok'\n", encoding="utf-8")
    issues = _issues_file(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return issues


def test_discover_creates_recoverable_run_and_artifacts(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)

    loaded = load_run(repo_root=tmp_path, run_id=run.run_id)
    artifact = tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "01_discover.json"

    assert loaded.run_id == run.run_id
    assert artifact.exists()
    assert "Add checkpoint docs" in artifact.read_text(encoding="utf-8")
    assert (artifact.parent / "01_discover_agent_prompt.md").exists()
    assert loaded.schema_version == 2
    assert loaded.base_commit_sha
    assert (artifact.parent / "metrics.json").exists()


def test_design_implement_and_draft_pr_use_previous_artifacts(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n\nUpdated.\n", encoding="utf-8")
    run = implement_stage(repo_root=tmp_path, run_id=run.run_id, agent_output="done")
    run = draft_pr_stage(repo_root=tmp_path, run_id=run.run_id)
    draft = tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "04_pr_draft.md"

    assert run.stage == "draft_pr"
    assert "**Problem**" in draft.read_text(encoding="utf-8")
    assert "**Notes for Reviewer**" in draft.read_text(encoding="utf-8")


def test_agent_reviews_are_attached_to_artifacts(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)

    attach_discover_agent_review(repo_root=tmp_path, run_id=run.run_id, review="Deep source analysis")
    design = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    attach_design_agent_review(repo_root=tmp_path, run_id=run.run_id, review="Concrete implementation plan")

    discover_md = (tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "01_discover.md").read_text(encoding="utf-8")
    design_md = (tmp_path / ".osc_agent" / "contribution_runs" / design.run_id / "02_design.md").read_text(encoding="utf-8")

    assert "Deep source analysis" in discover_md
    assert "Concrete implementation plan" in design_md


def test_prepare_implementation_happens_before_recording_result(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )

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


def test_execute_implementation_runs_ordered_substeps(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )
    calls = []

    def run_step(stage, prompt):
        calls.append((stage, prompt))
        if stage == "understanding":
            return "Scope confirmed. READY_TO_EDIT"
        return "implemented"

    execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)

    assert [stage for stage, _ in calls] == ["understanding", "edit"]
    assert "READY_TO_EDIT" in calls[1][1]
    report = (tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "03_implementation_report.md").read_text(
        encoding="utf-8"
    )
    assert "1 passed" in report


def test_execute_implementation_stops_without_ready_checkpoint(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={"allowed_files": ["README.md"], "files_to_modify": ["README.md"]},
    )
    calls = []

    def run_step(stage, prompt):
        calls.append(stage)
        return "The implementation boundary is unclear."

    try:
        execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    except ValueError as exc:
        assert "READY_TO_EDIT" in str(exc)
    else:
        raise AssertionError("implementation should stop before editing")

    assert calls == ["understanding"]


def test_failed_transition_persists_gate_reason_and_recovery_stage(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")

    try:
        prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    except ValueError as exc:
        assert "design scope contract is incomplete" in str(exc)
    else:
        raise AssertionError("invalid design must not transition to implementation")

    blocked = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert blocked.stage == "design"
    assert blocked.recovery_stage == "design"
    assert blocked.last_transition["status"] == "BLOCKED"


def test_resume_reuses_edit_checkpoint_and_reruns_only_failed_verification(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ["python -m pytest"],
        },
    )
    calls = []

    def run_step(stage, prompt):
        calls.append(stage)
        if stage == "understanding":
            return "READY_TO_EDIT"
        (tmp_path / "README.md").write_text("# Agent LLM Demo\n\nChanged.\n", encoding="utf-8")
        return "edited once"

    verification_runs = iter(
        [
            [{"command": "python -m pytest", "exit_code": 1}],
            [{"command": "python -m pytest", "exit_code": 0}],
        ]
    )
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.implementation.run_verification_commands",
        lambda *_args, **_kwargs: next(verification_runs),
    )

    first = execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    second = execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)

    assert first.final_status == RunStatus.FAILED_VALIDATION.value
    assert second.final_status == RunStatus.SUCCESS.value
    assert calls == ["understanding", "edit"]
    assert second.implementation_checkpoint["verification"]["status"] == "SUCCEEDED"


def test_controlled_verification_blocks_permission_sensitive_command(tmp_path):
    target = tmp_path / "README.md"
    target.write_text("keep\n", encoding="utf-8")

    results = run_verification_commands(tmp_path, ["Remove-Item README.md"])

    assert results[0]["exit_code"] == -2
    assert results[0]["permission"] == "ask"
    assert target.exists()


def test_issue_is_revalidated_before_implementation_transition(tmp_path, monkeypatch):
    issues_path = _prepare_repo(tmp_path)
    snapshot = json.loads(issues_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.discover._collect_issues",
        lambda **_kwargs: (snapshot["issues"], snapshot["comments_by_issue"], None),
    )
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo")
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={"allowed_files": ["README.md"], "files_to_modify": ["README.md"]},
    )
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.discover.fetch_issue",
        lambda *_args, **_kwargs: {"ok": True, "issue": {**snapshot["issues"][0], "state": "closed"}},
    )
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.discover.fetch_issue_comments",
        lambda *_args, **_kwargs: {"ok": True, "comments": []},
    )
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.discover.fetch_issue_activity",
        lambda *_args, **_kwargs: {"ok": True, "linked_pull_requests": []},
    )

    try:
        prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    except ValueError as exc:
        assert "Issue #7 is closed, assigned, claimed" in str(exc)
    else:
        raise AssertionError("unavailable issue must block implementation")


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
        assert "unsupported contribution run schema" in str(exc)
    else:
        raise AssertionError("design_stage should require discover artifact")


def test_load_run_rejects_path_traversal(tmp_path):
    try:
        load_run(repo_root=tmp_path, run_id="../../outside")
    except ValueError as exc:
        assert "run_id" in str(exc)
    else:
        raise AssertionError("run_id path traversal must be rejected")


def test_load_run_rejects_tampered_artifacts_path(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run_path = Path(run.artifacts_dir) / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["artifacts_dir"] = str(tmp_path.parent)
    run_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_run(repo_root=tmp_path, run_id=run.run_id)
    except ValueError as exc:
        assert "artifacts path" in str(exc)
    else:
        raise AssertionError("tampered artifacts path must be rejected")


def test_discover_blocks_dirty_source_repository(tmp_path):
    issues = _prepare_repo(tmp_path)
    (tmp_path / "agent.py").write_text("def run_agent():\n    return 'changed'\n", encoding="utf-8")

    try:
        discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    except ValueError as exc:
        assert "uncommitted changes" in str(exc)
    else:
        raise AssertionError("dirty repository must be blocked")


def test_scope_validation_rejects_unplanned_file(tmp_path):
    _prepare_repo(tmp_path)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")

    result = validate_implementation_scope(
        tmp_path,
        {
            "allowed_files": ["agent.py"],
            "allowed_new_dirs": ["tests"],
            "forbidden_paths": [".github/**"],
            "max_changed_files": 5,
            "max_diff_lines": 400,
        },
    )

    assert result["ok"] is False
    assert result["outside_scope"] == ["README.md"]


def test_design_validation_rejects_files_outside_repository(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("SECRET = 'outside'\n", encoding="utf-8")

    result = validate_design_files(
        tmp_path,
        {"files_to_modify": ["../outside.py"], "target_symbols": ["SECRET"]},
    )

    assert result["ok"] is False
    assert result["invalid_paths"] == ["../outside.py"]


def test_audited_test_waiver_allows_scoped_change_without_command(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={"allowed_files": ["README.md"], "files_to_modify": ["README.md"], "tests_to_run": []},
    )
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n\nDocs only.\n", encoding="utf-8")
    run = record_implementation_result(repo_root=tmp_path, run_id=run.run_id, agent_output="docs updated")

    assert run.final_status == RunStatus.FAILED_VALIDATION.value
    run = record_test_waiver(repo_root=tmp_path, run_id=run.run_id, reason="Documentation-only change")

    assert run.final_status == RunStatus.SUCCESS.value
    report = json.loads((Path(run.artifacts_dir) / "03_implementation.json").read_text(encoding="utf-8"))
    assert report["test_waiver"]["reason"] == "Documentation-only change"


def test_scope_budget_counts_untracked_file_lines(tmp_path):
    _prepare_repo(tmp_path)
    (tmp_path / "new_agent.py").write_text("\n".join("value = 1" for _ in range(401)) + "\n", encoding="utf-8")

    result = validate_implementation_scope(
        tmp_path,
        {
            "allowed_files": ["new_agent.py"],
            "allowed_new_dirs": [],
            "forbidden_paths": [".github/**"],
            "max_changed_files": 5,
            "max_diff_lines": 400,
        },
    )

    assert result["ok"] is False
    assert result["added_lines"] == 401
    assert any("diff line budget exceeded" in item for item in result["violations"])


def test_stage_artifact_hash_drift_marks_run_stale(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    artifact = Path(run.artifacts_dir) / "01_discover.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["top_directions"][0]["description"] = "tampered"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    try:
        design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Checkpoint docs")
    except ValueError as exc:
        assert "STALE_RUN" in str(exc)
        assert "stage artifact changed" in str(exc)
    else:
        raise AssertionError("tampered stage artifact must be rejected")


def test_atomic_json_write_preserves_previous_value_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    _write_raw_json(path, {"value": "before"})

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr("osc_agent.workflows.contribution.state.os.replace", fail_replace)
    try:
        _write_raw_json(path, {"value": "after"})
    except OSError:
        pass
    else:
        raise AssertionError("replace failure must propagate")

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "before"}


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
