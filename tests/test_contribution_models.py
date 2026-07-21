from __future__ import annotations

import pytest
from pydantic import ValidationError

from osc_agent.workflows.contribution.models import (
    AcceptanceCheck,
    BaselineCheck,
    ContributionRun,
    DesignContract,
    TransitionRecord,
    UnderstandingCheckpoint,
)


def _design_contract() -> dict:
    return {
        "selected_direction": "Fix retry",
        "contribution_spec": {
            "task_type": "behavior",
            "requirements": [{
                "id": "REQ-1",
                "text": "Retry transient failures",
                "source": "Issue #1",
                "source_excerpt": "Requests fail transiently",
                "kind": "behavior",
            }],
            "baseline_checks": [{
                "command": "pytest tests/test_retry.py",
                "expected_exit_codes": [1],
                "output_contains": "retry failed",
            }],
            "reproduction": {"mode": "existing", "command": "pytest tests/test_retry.py"},
        },
        "acceptance_checks": [{
            "criterion": "Retry test passes",
            "command": "pytest tests/test_retry.py",
            "requirement_ids": ["REQ-1"],
        }],
        "source_evidence": [{
            "file": "src/retry.py",
            "symbol": "retry",
            "line_range": [1, 2],
            "content_hash": "abc",
            "requirement_ids": ["REQ-1"],
        }],
    }


def test_automatic_acceptance_check_requires_command() -> None:
    with pytest.raises(ValidationError, match="must define command"):
        AcceptanceCheck(
            criterion="Tests pass",
            manual_check=False,
            requirement_ids=["REQ-1"],
        )


def test_failure_baseline_rejects_success_exit_code() -> None:
    with pytest.raises(ValidationError, match="non-zero"):
        BaselineCheck(
            command='python -c "print(\'import ok\')"',
            expected_exit_codes=[0],
            output_contains="import ok",
        )


def test_design_contract_rejects_unknown_requirement_reference() -> None:
    payload = _design_contract()
    payload["acceptance_checks"][0]["requirement_ids"] = ["REQ-2"]

    with pytest.raises(ValidationError, match="unknown requirements"):
        DesignContract.model_validate(payload)


def test_design_contract_rejects_uncovered_requirement() -> None:
    payload = _design_contract()
    payload["contribution_spec"]["requirements"].append({
        "id": "REQ-2",
        "text": "Preserve successful requests",
        "source": "Issue #1",
        "source_excerpt": "Successful requests remain unchanged",
        "kind": "behavior",
    })

    with pytest.raises(ValidationError, match="cover every requirement"):
        DesignContract.model_validate(payload)


def test_contribution_run_rejects_invalid_stage_and_schema() -> None:
    payload = {
        "run_id": "run_1",
        "repo_root": "C:/repo",
        "repo_url": "https://github.com/acme/repo",
        "stage": "unknown",
        "artifacts_dir": "C:/repo/.osc_agent/contribution_runs/run_1",
        "schema_version": 2,
        "stage_status": {
            "discover": "PENDING",
            "design": "PENDING",
            "implement": "PENDING",
            "draft_pr": "PENDING",
        },
    }

    with pytest.raises(ValidationError):
        ContributionRun.model_validate(payload)


def test_transition_record_preserves_persisted_from_key() -> None:
    transition = TransitionRecord.model_validate({
        "from": "design",
        "to": "implement",
        "status": "BLOCKED",
        "reason": "quality gate failed",
        "at": "2026-07-20T00:00:00+00:00",
    })

    assert transition.model_dump(mode="json", by_alias=True)["from"] == "design"


def test_understanding_checkpoint_requires_exact_ready_decision() -> None:
    with pytest.raises(ValidationError):
        UnderstandingCheckpoint.model_validate_json(
            '{"decision":"NOT READY_TO_EDIT","summary":"unclear",'
            '"requirement_ids":["REQ-1"],"files_to_modify":["src/a.py"],'
            '"unresolved_questions":[]}'
        )

    with pytest.raises(ValidationError, match="cannot contain unresolved questions"):
        UnderstandingCheckpoint.model_validate(
            {
                "decision": "READY_TO_EDIT",
                "summary": "Ready",
                "requirement_ids": ["REQ-1"],
                "files_to_modify": ["src/a.py"],
                "unresolved_questions": ["Which API should change?"],
            }
        )
