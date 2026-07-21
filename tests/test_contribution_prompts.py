from __future__ import annotations

from types import SimpleNamespace

from osc_agent.workflows.contribution.models import UnderstandingCheckpoint
from osc_agent.workflows.contribution.prompts import (
    _verification_diagnostics,
    build_edit_prompt,
    build_implementation_prompt,
)


def _design() -> dict:
    return {
        "selected_direction": "Fix retry behavior",
        "allowed_files": ["src/retry.py"],
        "allowed_new_dirs": ["tests/generated"],
        "forbidden_paths": [".github/**"],
        "max_changed_files": 2,
        "max_diff_lines": 80,
        "acceptance_checks": [
            {
                "criterion": "Retry regression passes",
                "command": "python -m pytest tests/test_retry.py",
                "manual_check": False,
                "requirement_ids": ["REQ-1"],
            }
        ],
        "contribution_spec": {
            "task_type": "behavior",
            "requirements": [{"id": "REQ-1", "text": "Retry once"}],
            "baseline_checks": [],
            "reproduction": {"mode": "generated_test", "test_files": ["tests/test_retry.py"]},
        },
    }


def test_implementation_overview_does_not_fabricate_understanding(tmp_path):
    run = SimpleNamespace(
        selected_direction="Fix retry behavior",
        repo_url="https://github.com/acme/demo",
        repo_root=str(tmp_path),
        worktree_root=None,
    )

    prompt = build_implementation_prompt(run, _design())

    assert "Prepared from saved workflow artifacts" not in prompt
    assert "No Understanding checkpoint has been completed" in prompt


def test_edit_prompt_uses_contract_and_real_reproduction_evidence(tmp_path):
    run = SimpleNamespace(
        selected_direction="Fix retry behavior",
        repo_url="https://github.com/acme/demo",
        repo_root=str(tmp_path),
        worktree_root=None,
    )
    understanding = UnderstandingCheckpoint(
        decision="READY_TO_EDIT",
        summary="Scope is concrete",
        requirement_ids=["REQ-1"],
        files_to_modify=["src/retry.py"],
        unresolved_questions=[],
    )

    prompt = build_edit_prompt(
        run,
        _design(),
        understanding,
        reproduction_evidence={"semantic_binding": {"matched_target_symbols": ["retry"]}},
    )

    assert '"allowed_new_dirs"' in prompt
    assert '"acceptance_checks"' in prompt
    assert '"semantic_binding"' in prompt
    assert "CONTRACT_UPDATE_REQUIRED" in prompt
    assert "unless the repository proves" not in prompt


def test_verification_diagnostics_rejects_external_artifact_path(tmp_path):
    outside = tmp_path.parent / "secret-verification.log"
    outside.write_text("secret", encoding="utf-8")

    diagnostics = _verification_diagnostics(
        tmp_path,
        [{"command": "pytest", "exit_code": 1, "artifact_path": str(outside)}],
    )

    assert "secret" not in diagnostics
    assert "INVALID_ARTIFACT_PATH" in diagnostics
