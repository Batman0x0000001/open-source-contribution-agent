from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from filelock import FileLock
import pytest

from osc_agent.workflows.contribution import (
    GateResult,
    attach_design_agent_review,
    attach_discover_human_review,
    build_design_review_prompt,
    configure_run,
    design_stage,
    discover_stage as _discover_stage,
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
from osc_agent.workflows.contribution.state import (
    acquire_run_lock,
    _evidence_file_hashes,
    _read_json,
    _require_consistent_run,
    _write_raw_json,
    _write_text,
    create_run,
    save_run,
)
from osc_agent.workflows.contribution.implementation import (
    _analyze_reproduction_semantics,
    _hash_test_files,
    _is_repairable_verification_failure,
)
from osc_agent.harness.contracts import RunStatus
from osc_agent.workflows.contribution.design import (
    _build_design_evidence,
    _build_requirements,
    _normalize_acceptance_checks,
)
from osc_agent.workflows.contribution.transitions import _begin_stage, _complete_stage
from stage_llm_fakes import FakeDiscoverClient, fake_settings, run_fake_discover


def discover_stage(**kwargs):
    return run_fake_discover(_discover_stage, **kwargs)


def _ready_understanding(*files_to_modify: str) -> str:
    return json.dumps(
        {
            "decision": "READY_TO_EDIT",
            "summary": "The saved contract is concrete and internally consistent.",
            "requirement_ids": ["REQ-1"],
            "files_to_modify": list(files_to_modify or ("README.md",)),
            "unresolved_questions": [],
        }
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
    assert loaded.schema_version == 3
    assert loaded.base_commit_sha
    assert (artifact.parent / "metrics.json").exists()


def test_discover_model_failure_persists_failed_stage(tmp_path):
    issues = _prepare_repo(tmp_path)

    class FailingMessages:
        def create(self, **_kwargs):
            raise RuntimeError("model unavailable")

    class FailingClient:
        messages = FailingMessages()

    with pytest.raises(RuntimeError, match="stage model request failed"):
        _discover_stage(
            repo_root=tmp_path,
            repo_url="https://github.com/acme/demo",
            issues_file=issues,
            client=FailingClient(),
            settings=fake_settings(),
        )

    runs_dir = tmp_path / ".osc_agent" / "contribution_runs"
    failed = load_run(repo_root=tmp_path, run_id=next(runs_dir.iterdir()).name)
    assert failed.stage_status["discover"] == "FAILED"
    assert failed.final_status == RunStatus.FAILED_TOOL.value


def test_selected_issue_uses_structured_direction_metadata(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.discover.score_candidate_issues",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.discover.run_discover_analysis",
        lambda *_args, **_kwargs: {
            "top_directions": [{
                "name": "Checkpoint documentation",
                "description": "Document the checkpoint workflow.",
                "source": "Selected issue",
                "source_kind": "issue",
                "issue_number": 7,
                "entry": "README.md",
                "effort": "small",
                "interview": "Shows documentation discipline",
                "risk": "low",
            }],
            "analysis_summary": "The issue is scoped and reviewable.",
            "architecture_insights": [],
        },
    )

    run = _discover_stage(
        repo_root=tmp_path,
        repo_url="https://github.com/acme/demo",
        issues_file=issues,
        client=object(),
        settings=fake_settings(),
    )
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="issue:7")

    assert run.selected_direction == "Checkpoint documentation"
    assert run.selected_direction_id == "issue:7"
    assert run.selected_issue_number == 7


def test_design_implement_and_draft_pr_use_previous_artifacts(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
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
    run = draft_pr_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        client=FakeDiscoverClient(),
        settings=fake_settings(),
    )
    draft = tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "04_pr_draft.md"
    draft_json = json.loads(draft.with_suffix(".json").read_text(encoding="utf-8"))

    assert run.stage == "draft_pr"
    assert "**Problem**" in draft.read_text(encoding="utf-8")
    assert "**Notes for Reviewer**" in draft.read_text(encoding="utf-8")
    assert draft_json["changed_files"] == ["README.md"]
    assert draft_json["generation_mode"] == "llm"
    assert draft_json["diff_hash"]


def test_human_discover_review_is_separate_from_stage_artifact(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)

    attach_discover_human_review(repo_root=tmp_path, run_id=run.run_id, review="Deep source analysis")
    design = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    attach_design_agent_review(repo_root=tmp_path, run_id=run.run_id, review="Concrete implementation plan")

    discover_review = json.loads(
        (tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "01_discover_human_review.json").read_text(
            encoding="utf-8"
        )
    )
    design_md = (tmp_path / ".osc_agent" / "contribution_runs" / design.run_id / "02_design.md").read_text(encoding="utf-8")

    assert discover_review["review"] == "Deep source analysis"
    assert "Concrete implementation plan" in design_md


def test_design_mutations_are_frozen_after_implementation_starts(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="issue:7")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )
    prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)

    with pytest.raises(ValueError, match="frozen after implementation starts"):
        update_design_contract(
            repo_root=tmp_path,
            run_id=run.run_id,
            updates={"max_diff_lines": 10},
        )
    with pytest.raises(ValueError, match="frozen after implementation starts"):
        attach_design_agent_review(repo_root=tmp_path, run_id=run.run_id, review="late review")
    with pytest.raises(ValueError, match="frozen after implementation starts"):
        configure_run(repo_root=tmp_path, run_id=run.run_id, settings=fake_settings())


def test_design_rejects_unknown_contribution_direction(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)

    try:
        design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Unrelated refactor")
    except ValueError as exc:
        assert "unknown contribution direction" in str(exc)
    else:
        raise AssertionError("design must stay linked to a discovered direction")


def test_design_contract_failure_marks_stage_failed(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.design.run_design_generation",
        lambda *_args, **_kwargs: {
            "requirements": [{"text": "Document checkpoints", "source_excerpt": "checkpoint docs"}],
            "acceptance_checks": [{
                "criterion": "Documentation is verified",
                "command": 'python -c "print(1)"',
                "manual_check": False,
                "requirement_ids": ["REQ-99"],
            }],
        },
    )

    with pytest.raises(ValueError, match="unknown requirements"):
        design_stage(
            repo_root=tmp_path,
            run_id=run.run_id,
            direction="issue:7",
            client=object(),
            settings=fake_settings(),
        )

    failed = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert failed.stage_status["design"] == "FAILED"
    assert failed.final_status == RunStatus.FAILED_VALIDATION.value


def test_design_model_failure_marks_stage_failed_tool(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.design.run_design_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        design_stage(
            repo_root=tmp_path,
            run_id=run.run_id,
            direction="issue:7",
            client=object(),
            settings=fake_settings(),
        )

    failed = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert failed.stage_status["design"] == "FAILED"
    assert failed.final_status == RunStatus.FAILED_TOOL.value


def test_design_model_receives_focused_symbol_evidence(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    captured: dict[str, object] = {}

    def capture_context(_client, _settings, context, _selected, **_kwargs):
        captured.update(context)
        return None

    monkeypatch.setattr(
        "osc_agent.workflows.contribution.design.run_design_generation",
        capture_context,
    )

    design_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        direction="issue:7",
        client=object(),
        settings=fake_settings(),
    )

    assert captured["selected_direction"]["id"] == "issue:7"
    assert captured["candidate_issues"][0]["number"] == 7
    assert "symbols" in captured["evidence_pack"]
    assert "repo_overview" not in captured


def test_design_builds_requirement_contract_from_selected_issue(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        direction="Issue #7: Add checkpoint docs",
    )

    design = json.loads((Path(run.artifacts_dir) / "02_design.json").read_text(encoding="utf-8"))

    assert design["contribution_spec"]["task_type"] == "docs"
    assert all(item["kind"] == "docs" for item in design["contribution_spec"]["requirements"])
    assert design["contribution_spec"]["requirements"][0]["id"] == "REQ-1"
    assert "checkpoint docs" in design["contribution_spec"]["requirements"][0]["text"].lower()


def test_design_evidence_uses_target_symbol_line_range(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        direction="Issue #7: Add checkpoint docs",
    )
    run = update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ['python -c "print(\'fixed\')"'],
            "baseline_checks": [
                {
                    "command": 'python -c "import sys; print(\'missing behavior\'); sys.exit(1)"',
                    "expected_exit_codes": [1],
                    "output_contains": "missing behavior",
                }
            ],
        },
    )

    design = json.loads((Path(run.artifacts_dir) / "02_design.json").read_text(encoding="utf-8"))
    evidence = design["source_evidence"]

    assert evidence[0]["symbol"] == "run_agent"
    assert evidence[0]["line_range"] == [1, 2]
    assert evidence[0]["requirement_ids"] == ["REQ-1"]


def test_behavior_implementation_reproduces_failure_before_edit_and_verifies_after(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_issue.py").write_text(
        "from agent import run_agent\n\n"
        "def test_issue_behavior():\n"
        "    assert run_agent() == 'fixed'\n",
        encoding="utf-8",
    )
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        direction="Issue #7: Add checkpoint docs",
    )
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_issue.py"],
            "baseline_checks": [
                {
                    "command": "python -m pytest -q tests/test_issue.py",
                    "expected_exit_codes": [1],
                    "output_contains": "1 failed",
                }
            ],
        },
    )
    calls = []

    def run_step(stage, _prompt):
        calls.append(stage)
        if stage == "understanding":
            return _ready_understanding("agent.py")
        (tmp_path / "agent.py").write_text("def run_agent():\n    return 'fixed'\n", encoding="utf-8")
        return "fixed behavior"

    completed = execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    report = json.loads((Path(run.artifacts_dir) / "03_implementation.json").read_text(encoding="utf-8"))

    assert completed.final_status == RunStatus.SUCCESS.value
    assert calls == ["understanding", "edit"]
    assert report["baseline_results"][0]["expected_failure_matched"] is True
    assert report["verification_results"][0]["exit_code"] == 0
    assert report["baseline_results"][0]["artifact_path"] != report["verification_results"][0]["artifact_path"]
    assert report["requirement_coverage"][0]["passed"] is True


def test_behavior_implementation_blocks_when_failure_cannot_be_reproduced(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_issue.py").write_text(
        "from agent import run_agent\n\n"
        "def test_issue_behavior():\n"
        "    assert run_agent() == 'ok'\n",
        encoding="utf-8",
    )
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        direction="Issue #7: Add checkpoint docs",
    )
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_issue.py"],
            "baseline_checks": [
                {
                    "command": "python -m pytest -q tests/test_issue.py",
                    "expected_exit_codes": [1],
                    "output_contains": "1 failed",
                }
            ],
        },
    )
    calls = []

    try:
        execute_implementation_stage(
            repo_root=tmp_path,
            run_id=run.run_id,
            run_step=lambda stage, _prompt: calls.append(stage) or _ready_understanding("agent.py"),
        )
    except ValueError as exc:
        assert "did not reproduce" in str(exc)
    else:
        raise AssertionError("implementation must stop before editing when the Issue cannot be reproduced")

    blocked = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert calls == []
    assert blocked.final_status == RunStatus.FAILED_VALIDATION.value


def test_behavior_implementation_can_generate_and_freeze_regression_test(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "allowed_new_dirs": ["tests"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_issue.py"],
            "reproduction_mode": "generated_test",
            "reproduction_test_files": ["tests/test_issue.py"],
        },
    )
    calls = []
    prompts = {}

    def run_step(stage, prompt):
        calls.append(stage)
        prompts[stage] = prompt
        if stage == "understanding":
            return _ready_understanding("agent.py")
        if stage == "reproduce":
            (tmp_path / "tests").mkdir()
            (tmp_path / "tests" / "test_issue.py").write_text(
                "from agent import run_agent\n\n"
                "def test_issue_behavior():\n"
                "    assert run_agent() == 'fixed'\n",
                encoding="utf-8",
            )
            return "added regression test"
        (tmp_path / "agent.py").write_text("def run_agent():\n    return 'fixed'\n", encoding="utf-8")
        return "fixed behavior"

    completed = execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    report = json.loads((Path(run.artifacts_dir) / "03_implementation.json").read_text(encoding="utf-8"))

    assert completed.final_status == RunStatus.SUCCESS.value
    assert calls == ["understanding", "reproduce", "edit"]
    assert report["baseline_results"][0]["expected_failure_matched"] is True
    assert report["reproduction_evidence"]["test_files"] == ["tests/test_issue.py"]
    assert report["reproduction_evidence"]["frozen_hashes"]["tests/test_issue.py"]
    assert report["reproduction_evidence"]["semantic_binding"]["matched_target_symbols"] == ["run_agent"]
    assert report["reproduction_evidence"]["semantic_binding"]["assertion_count"] == 1
    assert report["reproduction_evidence"]["semantic_binding"]["failure_references_test"] is True
    assert report["reproduction_validation"]["ok"] is True
    assert '"semantic_binding"' in prompts["edit"]
    assert '"acceptance_checks"' in prompts["edit"]
    assert '"allowed_new_dirs"' in prompts["edit"]
    assert "Python source files executed by pytest" in prompts["reproduce"]


def test_generated_reproduction_rejects_unrelated_failing_test(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "allowed_new_dirs": ["tests"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_issue.py"],
            "reproduction_mode": "generated_test",
            "reproduction_test_files": ["tests/test_issue.py"],
        },
    )
    calls = []

    def run_step(stage, _prompt):
        calls.append(stage)
        if stage == "understanding":
            return _ready_understanding("agent.py")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_issue.py").write_text(
            "def test_unrelated_failure():\n    assert 1 == 2\n",
            encoding="utf-8",
        )
        return "added an unrelated failing test"

    try:
        execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    except ValueError as exc:
        assert "does not bind the failure to an approved target symbol" in str(exc)
    else:
        raise AssertionError("an unrelated failing test must not authorize production edits")

    blocked = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert calls == ["understanding", "reproduce"]
    assert blocked.final_status == RunStatus.FAILED_VALIDATION.value


def test_failed_generated_reproduction_reports_reason_and_can_be_retried(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "allowed_new_dirs": ["tests"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_issue.py"],
            "reproduction_mode": "generated_test",
            "reproduction_test_files": ["tests/test_issue.py"],
        },
    )

    def invalid_step(stage, _prompt):
        if stage == "understanding":
            return _ready_understanding("agent.py")
        test_file = tmp_path / "tests" / "test_issue.py"
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text(
            "from agent import run_agent\n\ndef test_issue_behavior():\n    run_agent()\n    raise RuntimeError('boom')\n",
            encoding="utf-8",
        )
        return "added invalid regression test"

    with pytest.raises(ValueError, match="no supported assertion found"):
        execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=invalid_step)

    retry_prompts = {}

    def corrected_step(stage, prompt):
        retry_prompts[stage] = prompt
        if stage == "reproduce":
            (tmp_path / "tests" / "test_issue.py").write_text(
                "from agent import run_agent\n\ndef test_issue_behavior():\n    assert run_agent() == 'fixed'\n",
                encoding="utf-8",
            )
            return "corrected regression test"
        (tmp_path / "agent.py").write_text("def run_agent():\n    return 'fixed'\n", encoding="utf-8")
        return "fixed behavior"

    completed = execute_implementation_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        run_step=corrected_step,
    )

    assert completed.final_status == RunStatus.SUCCESS.value
    assert "no supported assertion found" in retry_prompts["reproduce"]


def test_behavior_implementation_rejects_changes_to_frozen_regression_test(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "allowed_new_dirs": ["tests"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_issue.py"],
            "reproduction_mode": "generated_test",
            "reproduction_test_files": ["tests/test_issue.py"],
        },
    )

    def run_step(stage, _prompt):
        if stage == "understanding":
            return _ready_understanding("agent.py")
        if stage == "reproduce":
            (tmp_path / "tests").mkdir()
            (tmp_path / "tests" / "test_issue.py").write_text(
                "from agent import run_agent\n\ndef test_issue_behavior():\n    assert run_agent() == 'fixed'\n",
                encoding="utf-8",
            )
            return "added regression test"
        (tmp_path / "agent.py").write_text("def run_agent():\n    return 'fixed'\n", encoding="utf-8")
        (tmp_path / "tests" / "test_issue.py").write_text(
            "def test_issue_behavior():\n    assert True\n",
            encoding="utf-8",
        )
        return "changed code and weakened test"

    try:
        execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    except ValueError as exc:
        assert "frozen regression test changed" in str(exc)
    else:
        raise AssertionError("implementation must not alter the test that established the failure")

    blocked = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert blocked.final_status == RunStatus.FAILED_VALIDATION.value


def test_generated_reproduction_cannot_be_bypassed_by_direct_result_recording(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "allowed_new_dirs": ["tests"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest -q tests/test_issue.py"],
            "reproduction_mode": "generated_test",
            "reproduction_test_files": ["tests/test_issue.py"],
        },
    )
    prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_issue.py").write_text(
        "def test_unrelated_success():\n    assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "agent.py").write_text("def run_agent():\n    return 'fixed'\n", encoding="utf-8")

    completed = record_implementation_result(repo_root=tmp_path, run_id=run.run_id, agent_output="claimed fix")
    report = json.loads((Path(run.artifacts_dir) / "03_implementation.json").read_text(encoding="utf-8"))

    assert completed.final_status == RunStatus.FAILED_VALIDATION.value
    assert report["reproduction_validation"]["ok"] is False


def test_prepare_implementation_happens_before_recording_result(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
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
    assert "No Understanding checkpoint has been completed" in prompt

    record_implementation_result(
        repo_root=tmp_path,
        run_id=run.run_id,
        agent_output="pytest tests/test_demo.py\n1 passed",
    )

    report = report_path.read_text(encoding="utf-8")
    assert "1 passed" in report
    assert "Implementation has not run yet" not in report


def test_record_result_does_not_implicitly_start_implementation(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="issue:7")

    with pytest.raises(ValueError, match="must be RUNNING"):
        record_implementation_result(repo_root=tmp_path, run_id=run.run_id, agent_output="unexpected")

    unchanged = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert unchanged.stage == "design"
    assert unchanged.stage_status["implement"] == "PENDING"


def test_prepare_resume_rejects_checkpoint_divergence(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="issue:7")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )
    run, _ = prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    run.implementation_checkpoint = {
        "understanding": {"status": "SUCCEEDED", "output": _ready_understanding()}
    }
    save_run(run)

    with pytest.raises(ValueError, match="checkpoint state diverged"):
        prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)


def test_execute_implementation_runs_ordered_substeps(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
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
            return _ready_understanding()
        return "implemented"

    execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)

    assert [stage for stage, _ in calls] == ["understanding", "edit"]
    assert '"requirement_ids":["REQ-1"]' in calls[0][1]
    assert '"files_to_modify":["README.md"]' in calls[0][1]
    assert "path/from/contract" not in calls[0][1]
    assert '"decision": "READY_TO_EDIT"' in calls[1][1]
    report = (tmp_path / ".osc_agent" / "contribution_runs" / run.run_id / "03_implementation_report.md").read_text(
        encoding="utf-8"
    )
    assert "1 passed" in report


def test_execute_implementation_stops_without_ready_checkpoint(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
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
        calls.append(stage)
        return "The implementation boundary is unclear."

    try:
        execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    except ValueError as exc:
        assert "invalid understanding checkpoint" in str(exc)
        assert "root: Invalid JSON" in str(exc)
    else:
        raise AssertionError("implementation should stop before editing")

    assert calls == ["understanding", "understanding"]


def test_understanding_retries_invalid_format_before_editing(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )
    calls: list[str] = []

    def run_step(stage, _prompt):
        calls.append(stage)
        if calls == ["understanding"]:
            return "I am ready to edit."
        if stage == "understanding":
            return _ready_understanding()
        (tmp_path / "README.md").write_text("# Agent LLM Demo\n\nUpdated.\n", encoding="utf-8")
        return "implemented"

    completed = execute_implementation_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        run_step=run_step,
    )

    assert calls == ["understanding", "understanding", "edit"]
    assert completed.stage_status["implement"] == "SUCCEEDED"


def test_failed_understanding_can_be_explicitly_retried_before_edits(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )

    with pytest.raises(ValueError, match="invalid understanding checkpoint"):
        execute_implementation_stage(
            repo_root=tmp_path,
            run_id=run.run_id,
            run_step=lambda _stage, _prompt: "invalid",
        )

    def successful_step(stage, _prompt):
        if stage == "understanding":
            return _ready_understanding()
        (tmp_path / "README.md").write_text("# Agent LLM Demo\n\nUpdated.\n", encoding="utf-8")
        return "implemented"

    completed = execute_implementation_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        run_step=successful_step,
    )

    assert completed.stage_status["implement"] == "SUCCEEDED"


def test_edit_stops_when_agent_requests_contract_update(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
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

    def run_step(stage, _prompt):
        calls.append(stage)
        return _ready_understanding() if stage == "understanding" else "CONTRACT_UPDATE_REQUIRED"

    with pytest.raises(ValueError, match="CONTRACT_UPDATE_REQUIRED"):
        execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)

    blocked = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert calls == ["understanding", "edit"]
    assert blocked.final_status == RunStatus.BLOCKED_NEEDS_USER.value
    assert blocked.stage_status["implement"] == "FAILED"


def test_failed_transition_persists_gate_reason_and_recovery_stage(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")

    try:
        prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    except ValueError as exc:
        assert "design scope contract is incomplete" in str(exc)
    else:
        raise AssertionError("invalid design must not transition to implementation")

    blocked = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert blocked.stage == "design"
    assert blocked.recovery_stage == "design"
    assert blocked.last_transition is not None
    assert blocked.last_transition.status == "BLOCKED"


def test_failed_verification_triggers_repair_without_manual_resume(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
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
    repair_prompts = []

    def run_step(stage, prompt):
        calls.append(stage)
        if stage == "understanding":
            return _ready_understanding()
        if stage == "repair":
            repair_prompts.append(prompt)
        content = "Changed." if stage == "edit" else "Repaired."
        (tmp_path / "README.md").write_text(f"# Agent LLM Demo\n\n{content}\n", encoding="utf-8")
        return content

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

    completed = execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)

    assert completed.final_status == RunStatus.SUCCESS.value
    assert calls == ["understanding", "edit", "repair"]
    assert completed.implementation_checkpoint["verification"]["status"] == "SUCCEEDED"
    assert len(completed.implementation_checkpoint["repair_attempts"]) == 1
    assert completed.implementation_checkpoint["repair_attempts"][0]["status"] == "SUCCEEDED"
    assert '"exit_code": 1' in repair_prompts[0]
    assert "Changed." in repair_prompts[0]


def test_failed_verification_stops_after_repair_limit(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ["python -m pytest"],
        },
    )
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.implementation.run_verification_commands",
        lambda *_args, **_kwargs: [{"command": "python -m pytest", "exit_code": 1}],
    )
    calls = []

    def run_step(stage, _prompt):
        calls.append(stage)
        if stage == "understanding":
            return _ready_understanding()
        (tmp_path / "README.md").write_text(
            f"# Agent LLM Demo\n\nAttempt {len(calls)}.\n",
            encoding="utf-8",
        )
        return f"attempt {len(calls)}"

    completed = execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)
    with pytest.raises(ValueError, match="resume only while the stage is RUNNING"):
        execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)

    assert completed.final_status == RunStatus.FAILED_VALIDATION.value
    assert calls == ["understanding", "edit", "repair", "repair", "repair"]
    assert len(completed.implementation_checkpoint["repair_attempts"]) == 3
    assert all(item["status"] == "FAILED" for item in completed.implementation_checkpoint["repair_attempts"])
    assert completed.implementation_checkpoint["edit"]["status"] == "NEEDS_REPAIR"


def test_repair_loop_distinguishes_test_failures_from_execution_failures():
    assert _is_repairable_verification_failure(
        [{"command": "python -m pytest tests/test_agent.py", "exit_code": 1}]
    ) is True
    assert _is_repairable_verification_failure(
        [{"command": "python -m pytest tests/test_agent.py", "exit_code": 2}]
    ) is False
    assert _is_repairable_verification_failure(
        [{"command": "python -m pytest tests/test_agent.py", "exit_code": -2}]
    ) is False


def test_behavior_resume_reuses_successful_pre_change_baseline(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(
        repo_root=tmp_path,
        run_id=run.run_id,
        direction="Issue #7: Add checkpoint docs",
    )
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "task_type": "behavior",
            "allowed_files": ["agent.py"],
            "files_to_modify": ["agent.py"],
            "target_symbols": ["run_agent"],
            "tests_to_run": ["python -m pytest tests/test_agent.py"],
            "baseline_checks": [
                {
                    "command": "python -m pytest tests/test_agent.py",
                    "expected_exit_codes": [1],
                    "output_contains": "expected failure",
                }
            ],
        },
    )
    baseline_calls = []
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.implementation.run_baseline_checks",
        lambda *_args, **_kwargs: baseline_calls.append("baseline") or [
            {
                "command": "python -m pytest tests/test_agent.py",
                "exit_code": 1,
                "expected_failure_matched": True,
            }
        ],
    )
    verification_runs = iter(
        [
            [{"command": "python -m pytest tests/test_agent.py", "exit_code": 1}],
            [{"command": "python -m pytest tests/test_agent.py", "exit_code": 0}],
        ]
    )
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.implementation.run_verification_commands",
        lambda *_args, **_kwargs: next(verification_runs),
    )
    edit_calls = []

    def run_step(stage, _prompt):
        edit_calls.append(stage)
        if stage == "understanding":
            return _ready_understanding("agent.py")
        (tmp_path / "agent.py").write_text("def run_agent():\n    return 'fixed'\n", encoding="utf-8")
        return "fixed"

    completed = execute_implementation_stage(repo_root=tmp_path, run_id=run.run_id, run_step=run_step)

    assert completed.final_status == RunStatus.SUCCESS.value
    assert baseline_calls == ["baseline"]
    assert edit_calls == ["understanding", "edit", "repair"]


def test_controlled_verification_blocks_permission_sensitive_command(tmp_path):
    target = tmp_path / "README.md"
    target.write_text("keep\n", encoding="utf-8")

    results = run_verification_commands(tmp_path, ["Remove-Item README.md"])

    assert results[0]["exit_code"] == -2
    assert results[0]["permission"] == "ask"
    assert target.exists()


def test_verification_prefers_worktree_src_over_editable_install(tmp_path, monkeypatch):
    installed_src = tmp_path / "source-repository" / "src"
    worktree = tmp_path / "worktree"
    worktree_src = worktree / "src"
    (installed_src / "demo_package").mkdir(parents=True)
    (worktree_src / "demo_package").mkdir(parents=True)
    (installed_src / "demo_package" / "__init__.py").write_text("VALUE = 'old'\n", encoding="utf-8")
    (worktree_src / "demo_package" / "__init__.py").write_text("VALUE = 'worktree'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(installed_src))

    command = (
        f'"{sys.executable}" -c "import demo_package; '
        "assert demo_package.VALUE == 'worktree', demo_package.__file__\""
    )
    result = run_verification_commands(worktree, [command])[0]

    assert result["exit_code"] == 0


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


def test_allowed_new_dirs_does_not_authorize_existing_files(tmp_path):
    _prepare_repo(tmp_path)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    existing = tests_dir / "existing.py"
    existing.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/existing.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add existing test"], cwd=tmp_path, check=True, capture_output=True)
    existing.write_text("VALUE = 2\n", encoding="utf-8")
    (tests_dir / "new_test.py").write_text("def test_new():\n    assert True\n", encoding="utf-8")

    result = validate_implementation_scope(
        tmp_path,
        {
            "allowed_files": [],
            "allowed_new_dirs": ["tests"],
            "forbidden_paths": [".github/**"],
            "max_changed_files": 5,
            "max_diff_lines": 400,
        },
    )

    assert result["ok"] is False
    assert result["added_files"] == ["tests/new_test.py"]
    assert result["modified_files"] == ["tests/existing.py"]
    assert result["outside_scope"] == ["tests/existing.py"]


def test_scope_rejects_deletion_rename_and_binary_changes(tmp_path):
    _prepare_repo(tmp_path)
    binary = tmp_path / "asset.bin"
    binary.write_bytes(b"original\n")
    subprocess.run(["git", "add", "asset.bin"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add binary target"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").unlink()
    subprocess.run(["git", "mv", "agent.py", "renamed.py"], cwd=tmp_path, check=True)
    binary.write_bytes(b"\x00binary\n")

    result = validate_implementation_scope(
        tmp_path,
        {
            "allowed_files": ["README.md", "agent.py", "renamed.py", "asset.bin"],
            "allowed_new_dirs": [],
            "forbidden_paths": [".github/**"],
            "max_changed_files": 5,
            "max_diff_lines": 400,
        },
    )

    assert result["ok"] is False
    assert result["deleted_files"] == ["README.md"]
    assert result["renamed_files"] == [{"from": "agent.py", "to": "renamed.py"}]
    assert result["binary_files"] == ["asset.bin"]


def test_scope_contract_rejects_invalid_types_and_budgets(tmp_path):
    _prepare_repo(tmp_path)

    with pytest.raises(ValueError):
        validate_implementation_scope(
            tmp_path,
            {
                "allowed_files": "agent.py",
                "allowed_new_dirs": [],
                "forbidden_paths": ".github/**",
                "max_changed_files": 0,
                "max_diff_lines": "400",
            },
        )


def test_scope_validation_ignores_test_runtime_caches(tmp_path):
    _prepare_repo(tmp_path)
    (tmp_path / "agent.py").write_text("def run_agent():\n    return 'fixed'\n", encoding="utf-8")
    cache = tmp_path / "tests" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "test_agent.pyc").write_bytes(b"cache")
    pytest_cache = tmp_path / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "README.md").write_text("generated\n" * 500, encoding="utf-8")

    result = validate_implementation_scope(
        tmp_path,
        {
            "allowed_files": ["agent.py"],
            "allowed_new_dirs": [],
            "forbidden_paths": [".github/**"],
            "max_changed_files": 5,
            "max_diff_lines": 400,
        },
    )

    assert result["ok"] is True
    assert result["changed_files"] == ["agent.py"]
    assert result["added_lines"] == 1


def test_design_validation_rejects_files_outside_repository(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("SECRET = 'outside'\n", encoding="utf-8")

    result = validate_design_files(
        tmp_path,
        {"files_to_modify": ["../outside.py"], "target_symbols": ["SECRET"]},
    )

    assert result["ok"] is False
    assert result["invalid_paths"] == ["../outside.py"]


def test_design_validation_supports_qualified_methods(tmp_path):
    (tmp_path / "agent.py").write_text(
        "class DefaultAgent:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def run(self):\n"
        "        pass\n"
        "    def step(self):\n"
        "        pass\n"
        "    def add_messages(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    symbols = [
        "DefaultAgent.__init__",
        "DefaultAgent.run",
        "DefaultAgent.step",
        "DefaultAgent.add_messages",
        "DefaultAgent.missing",
    ]

    result = validate_design_files(
        tmp_path,
        {"files_to_modify": ["agent.py"], "target_symbols": symbols},
    )

    assert result["ok"] is False
    assert result["missing_symbols"] == ["DefaultAgent.missing"]


def test_audited_test_waiver_allows_scoped_change_without_command(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={"allowed_files": ["README.md"], "files_to_modify": ["README.md"], "tests_to_run": []},
    )
    prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    (tmp_path / "README.md").write_text("# Agent LLM Demo\n\nDocs only.\n", encoding="utf-8")
    run = record_test_waiver(repo_root=tmp_path, run_id=run.run_id, reason="Documentation-only change")

    assert run.final_status == RunStatus.SUCCESS.value
    report = json.loads((Path(run.artifacts_dir) / "03_implementation.json").read_text(encoding="utf-8"))
    assert report["test_waiver"]["reason"] == "Documentation-only change"


def test_test_waiver_recomputes_scope_and_does_not_restart_failed_stage(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="issue:7")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={"allowed_files": ["README.md"], "files_to_modify": ["README.md"], "tests_to_run": []},
    )
    prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    (tmp_path / "agent.py").write_text("def run_agent():\n    return 'outside scope'\n", encoding="utf-8")

    failed = record_test_waiver(repo_root=tmp_path, run_id=run.run_id, reason="docs only")
    report = json.loads((Path(run.artifacts_dir) / "03_implementation.json").read_text(encoding="utf-8"))
    assert failed.final_status == RunStatus.FAILED_VALIDATION.value
    assert report["scope_validation"]["outside_scope"] == ["agent.py"]

    with pytest.raises(ValueError, match="must be RUNNING"):
        record_test_waiver(repo_root=tmp_path, run_id=run.run_id, reason="retry waiver")


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
        design_stage(repo_root=tmp_path, run_id=run.run_id, direction="Issue #7: Add checkpoint docs")
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


def test_save_failure_does_not_advance_in_memory_revision(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    original_revision = run.revision

    monkeypatch.setattr(
        "osc_agent.workflows.contribution.state.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("injected replace failure")),
    )

    with pytest.raises(OSError, match="injected replace failure"):
        save_run(run)

    assert run.revision == original_revision


def test_save_run_uses_independent_state_lock(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    state_lock = Path(run.artifacts_dir) / "state.lock"
    monkeypatch.setattr("osc_agent.workflows.contribution.state.STATE_LOCK_TIMEOUT_SECONDS", 0)

    with FileLock(state_lock):
        with pytest.raises(ValueError, match="state is busy"):
            save_run(run)


def test_run_json_readers_reject_corrupt_or_non_object_state(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    state_path = Path(run.artifacts_dir) / "run.json"

    state_path.write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid contribution run state"):
        load_run(repo_root=tmp_path, run_id=run.run_id)
    with pytest.raises(ValueError, match="invalid contribution run state"):
        save_run(run)

    state_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root must be an object"):
        load_run(repo_root=tmp_path, run_id=run.run_id)


def test_artifact_reader_requires_json_object(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    artifact = Path(run.artifacts_dir) / "invalid.json"
    artifact.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON root must be an object"):
        _read_json(run, "invalid.json")
    with pytest.raises(ValueError, match="path escapes repository"):
        _read_json(run, "../run.json")


def test_evidence_hashes_use_safe_paths_and_original_bytes(tmp_path):
    _prepare_repo(tmp_path)
    binary = tmp_path / "evidence.bin"
    content = b"\xff\xfe\x00evidence\r\n"
    binary.write_bytes(content)

    hashes = _evidence_file_hashes(tmp_path, {"file": "evidence.bin"})

    assert hashes["evidence.bin"] == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="path escapes repository"):
        _evidence_file_hashes(tmp_path, {"file": "../outside.txt"})


def test_atomic_text_write_preserves_previous_value_on_replace_failure(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    path = Path(run.artifacts_dir) / "report.md"
    _write_text(run, "report.md", "before")
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.state.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("injected replace failure")),
    )

    with pytest.raises(OSError, match="injected replace failure"):
        _write_text(run, "report.md", "after")

    assert path.read_text(encoding="utf-8") == "before\n"


def test_temp_cleanup_error_does_not_hide_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    _write_raw_json(path, {"value": "before"})
    original_unlink = Path.unlink

    monkeypatch.setattr(
        "osc_agent.workflows.contribution.state.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    def fail_temp_cleanup(target, *args, **kwargs):
        if target.suffix == ".tmp":
            raise OSError("cleanup failed")
        return original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_cleanup)

    with pytest.raises(OSError, match="replace failed"):
        _write_raw_json(path, {"value": "after"})


def test_stale_reason_is_preserved_when_status_save_fails(tmp_path, monkeypatch):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run.base_commit_sha = "0" * 40
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.state.save_run",
        lambda _run: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(ValueError, match="repository HEAD changed.*failed to persist stale status"):
        _require_consistent_run(run, tmp_path)


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


def test_enumerated_issue_builds_atomic_requirements():
    discover = {
        "candidate_issues": [{
            "number": 7,
            "title": "Fix retries",
            "body": "- Retry timeouts once\n- Preserve the original exception",
        }]
    }

    requirements = _build_requirements(discover, "Fix retries", issue_number=7)

    assert [item["id"] for item in requirements] == ["REQ-1", "REQ-2"]
    assert requirements[1]["source_excerpt"] == "Preserve the original exception"


def test_multiple_requirements_require_explicit_acceptance_mapping():
    with pytest.raises(ValueError, match="missing: REQ-1, REQ-2"):
        _normalize_acceptance_checks(
            None,
            [],
            ["python -m pytest tests/test_retry.py"],
            ["REQ-1", "REQ-2"],
            "behavior",
        )


def test_acceptance_mapping_reports_partially_uncovered_requirements():
    with pytest.raises(ValueError, match="missing: REQ-2"):
        _normalize_acceptance_checks(
            [{
                "criterion": "Retry succeeds",
                "command": "python -m pytest tests/test_retry.py",
                "manual_check": False,
                "requirement_ids": ["REQ-1"],
            }],
            [],
            [],
            ["REQ-1", "REQ-2"],
            "behavior",
        )


def test_design_evidence_rejects_missing_paths_and_symbols(tmp_path):
    _prepare_repo(tmp_path)

    with pytest.raises(ValueError, match="allowed_files may contain only existing repository files"):
        _build_design_evidence(tmp_path, ["missing.py"], [], ["REQ-1"])
    with pytest.raises(ValueError, match="target symbols not found: missing_symbol"):
        _build_design_evidence(tmp_path, ["agent.py"], ["missing_symbol"], ["REQ-1"])


def test_design_evidence_supports_class_qualified_method(tmp_path):
    (tmp_path / "agent.py").write_text(
        "class DefaultAgent:\n"
        "    def run(self):\n"
        "        return 'ok'\n",
        encoding="utf-8",
    )

    evidence = _build_design_evidence(
        tmp_path,
        ["agent.py"],
        ["DefaultAgent.run"],
        ["REQ-1"],
    )

    assert evidence[0]["symbol"] == "DefaultAgent.run"
    assert evidence[0]["line_range"] == [2, 3]


def test_reproduction_semantics_matches_class_qualified_method(tmp_path):
    test_file = tmp_path / "tests" / "test_agent.py"
    test_file.parent.mkdir()
    test_file.write_text(
        "def test_run(agent):\n"
        "    assert agent.run() == 'ok'\n",
        encoding="utf-8",
    )

    result = _analyze_reproduction_semantics(
        repo_root=tmp_path,
        test_files=["tests/test_agent.py"],
        target_symbols=["DefaultAgent.run"],
        failure_output="FAILED tests/test_agent.py::test_run",
        requirement_ids=["REQ-1"],
    )

    assert result["ok"] is True
    assert result["matched_target_symbols"] == ["DefaultAgent.run"]


def test_run_lock_rejects_concurrent_execution(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)

    with acquire_run_lock(repo_root=tmp_path, run_id=run.run_id):
        try:
            with acquire_run_lock(repo_root=tmp_path, run_id=run.run_id):
                raise AssertionError("second lock must not be acquired")
        except ValueError as exc:
            assert "already executing" in str(exc)


def test_public_implementation_writers_honor_run_lock(tmp_path):
    issues = _prepare_repo(tmp_path)
    run = discover_stage(repo_root=tmp_path, repo_url="https://github.com/acme/demo", issues_file=issues)
    run = design_stage(repo_root=tmp_path, run_id=run.run_id, direction="issue:7")
    update_design_contract(
        repo_root=tmp_path,
        run_id=run.run_id,
        updates={
            "allowed_files": ["README.md"],
            "files_to_modify": ["README.md"],
            "tests_to_run": ['python -c "print(1)"'],
        },
    )

    with acquire_run_lock(repo_root=tmp_path, run_id=run.run_id):
        with pytest.raises(ValueError, match="already executing"):
            prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
        with pytest.raises(ValueError, match="already executing"):
            implement_stage(repo_root=tmp_path, run_id=run.run_id)

    prepare_implementation_stage(repo_root=tmp_path, run_id=run.run_id)
    with acquire_run_lock(repo_root=tmp_path, run_id=run.run_id):
        with pytest.raises(ValueError, match="already executing"):
            record_implementation_result(repo_root=tmp_path, run_id=run.run_id, agent_output="done")
        with pytest.raises(ValueError, match="already executing"):
            record_test_waiver(repo_root=tmp_path, run_id=run.run_id, reason="docs only")


def test_test_file_hash_uses_safe_path_and_original_bytes(tmp_path):
    _prepare_repo(tmp_path)
    test_file = tmp_path / "tests" / "test_bytes.py"
    test_file.parent.mkdir()
    content = b"def test_bytes():\r\n    assert True\r\n"
    test_file.write_bytes(content)

    hashes = _hash_test_files(tmp_path, ["tests/test_bytes.py"])

    assert hashes["tests/test_bytes.py"] == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="path escapes repository"):
        _hash_test_files(tmp_path, ["../outside_test.py"])


def test_complete_stage_persists_transition_immediately(tmp_path):
    _prepare_repo(tmp_path)
    run = create_run(repo_root=tmp_path, repo_url="https://github.com/acme/demo")
    _begin_stage(run, "discover", tmp_path)

    _complete_stage(run, "discover", success=True)

    loaded = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert loaded.stage_status["discover"] == "SUCCEEDED"
    assert loaded.recovery_stage is None
    assert loaded.last_transition is not None
    assert loaded.last_transition.status == "SUCCEEDED"


def test_forward_transition_requires_current_stage_success(tmp_path, monkeypatch):
    _prepare_repo(tmp_path)
    run = create_run(repo_root=tmp_path, repo_url="https://github.com/acme/demo")
    monkeypatch.setattr(
        "osc_agent.workflows.contribution.transitions._transition_gate",
        lambda *_args, **_kwargs: GateResult(True, "test gate passed"),
    )
    _begin_stage(run, "discover", tmp_path)

    with pytest.raises(ValueError, match="illegal contribution transition"):
        _begin_stage(run, "design", tmp_path)

    _complete_stage(run, "discover", success=False)
    with pytest.raises(ValueError, match="illegal contribution transition"):
        _begin_stage(run, "design", tmp_path)

    # 失败阶段仍允许原阶段重试，但不能被跨过。
    _begin_stage(run, "discover", tmp_path)
    _complete_stage(run, "discover", success=True)
    _begin_stage(run, "design", tmp_path)

    loaded = load_run(repo_root=tmp_path, run_id=run.run_id)
    assert loaded.stage == "design"
    assert loaded.stage_status["discover"] == "SUCCEEDED"
    assert loaded.stage_status["design"] == "RUNNING"
    build_design_review_prompt,
