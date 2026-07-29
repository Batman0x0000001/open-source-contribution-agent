"""验证Bot 数据模型的契约、边界条件与回归行为。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest
from pydantic import ValidationError

from osc_agent.bot.config import load_repository_catalog
from osc_agent.bot.domain.artifacts import DeliveryDraft, IssuePlanArtifact
from osc_agent.bot.domain.execution import (
    BotApproval,
    ExecutionContract,
    plan_evidence_hash,
    validate_implementation_approval,
)
from osc_agent.bot.domain.repositories import RepositoryBotConfig


SHA = "a" * 40
FINGERPRINT = "b" * 64
IMAGE_ID = "sha256:" + "c" * 64


@pytest.mark.parametrize(
    "image",
    ["worker:latest", "sha256:abc", "sha256:" + "A" * 64],
)
def test_repository_config_requires_an_immutable_image_id(image: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        RepositoryBotConfig(
            image=image,
            validation_commands=("python -m pytest",),
        )


def test_repository_config_requires_a_recognized_test_at_loading_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repositories.yml"
    path.write_text(
        f"repositories:\n  owner/repo:\n    image: {IMAGE_ID}\n"
        "    validation_commands: [python -m pip check]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="recognized test"):
        load_repository_catalog(path)
    config = RepositoryBotConfig(
        image=IMAGE_ID,
        validation_commands=("python -m pytest", "python -m pip check"),
    )
    assert config.validation_commands[0] == "python -m pytest"


def test_repository_catalog_is_strict_and_rejects_escape(tmp_path: Path) -> None:
    path = tmp_path / "repositories.yml"
    path.write_text(
        f"repositories:\n  owner/repo:\n    image: {IMAGE_ID}\n    validation_commands: [python -m pytest]\n",
        encoding="utf-8",
    )
    assert "owner/repo" in load_repository_catalog(path).repositories
    with pytest.raises(ValidationError, match="repository-relative"):
        RepositoryBotConfig(
            image=IMAGE_ID,
            validation_commands=("python -m pytest",),
            denied_paths=("../secret",),
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RepositoryBotConfig.model_validate({
            "image": IMAGE_ID,
            "validation_commands": ["python -m pytest"],
            "auto_merge": True,
        })


def test_execution_contract_hash_is_canonical_and_secret_free() -> None:
    values = {
        "repository_id": 1, "repository_full_name": "owner/repo", "installation_id": 2,
        "base_branch": "main", "base_sha": SHA, "issue_number": 3,
        "issue_input_hash": "d" * 64, "model_id": "model",
        "plan_allowed_tools": frozenset({"read_file"}),
        "implementation_allowed_tools": frozenset({"read_file", "edit_file"}),
        "validation_commands": ("python -m pytest",), "denied_paths": (".git/**",),
        "max_changed_files": 10, "max_patch_bytes": 1000, "image_id": IMAGE_ID,
        "command_timeout_seconds": 60, "container_cpus": 1.0,
        "container_memory": "1g", "container_pids": 64,
    }
    first = ExecutionContract.model_validate(values)
    second = ExecutionContract.model_validate(dict(reversed(list(values.items()))))
    assert first.contract_hash == second.contract_hash
    assert "secret" not in first.model_dump_json().lower()


def test_execution_contract_hash_is_stable_across_process_hash_seeds() -> None:
    script = """
from osc_agent.bot.domain.execution import ExecutionContract

contract = ExecutionContract(
    repository_id=1,
    repository_full_name="owner/repo",
    installation_id=2,
    base_branch="main",
    base_sha="a" * 40,
    issue_number=3,
    issue_input_hash="d" * 64,
    model_id="model",
    plan_allowed_tools=frozenset({
        "read_file", "glob", "grep", "git_status", "git_diff", "git_log",
        "read_tool_result", "agent", "submit_issue_plan",
    }),
    implementation_allowed_tools=frozenset({
        "read_file", "glob", "grep", "bash", "git_status", "git_diff",
        "git_log", "read_skill_resource", "write_file", "edit_file",
        "read_tool_result", "agent", "submit_delivery_draft",
    }),
    validation_commands=("python -m pytest",),
    denied_paths=(".git/**",),
    max_changed_files=10,
    max_patch_bytes=1000,
    image_id="sha256:" + "c" * 64,
    command_timeout_seconds=60,
    container_cpus=1.0,
    container_memory="1g",
    container_pids=64,
)
print(contract.contract_hash)
"""
    hashes = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[2],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("1", "2", "3", "4")
    }

    assert len(hashes) == 1


def test_plan_status_and_delivery_contracts_are_hard_validated() -> None:
    with pytest.raises(ValidationError, match="ready plan"):
        IssuePlanArtifact(
            status="ready",
            base_sha=SHA,
            summary="summary",
            plan_markdown="plan",
            unresolved_questions=["question"],
        )
    with pytest.raises(ValidationError, match="blocked plan"):
        IssuePlanArtifact(
            status="blocked",
            base_sha=SHA,
            summary="summary",
            plan_markdown="plan",
        )
    draft = DeliveryDraft(
        title="Fix issue",
        body="Evidence-backed draft",
        commit_message="fix: issue",
        issue_number=7,
        base_sha=SHA,
        snapshot_fingerprint=FINGERPRINT,
        test_summary="pytest passed",
        verification_summary="Verify PASS",
    )
    assert draft.evidence_type == "delivery_draft"


def test_implementation_approval_is_bound_to_plan_and_expiry() -> None:
    plan = IssuePlanArtifact(
        status="ready",
        base_sha=SHA,
        summary="summary",
        plan_markdown="plan",
    )
    approval = BotApproval(
        approval_id=str(uuid4()),
        job_id=str(uuid4()),
        actor_id=1,
        actor_login="maintainer",
        evidence_hash=plan_evidence_hash(plan),
        execution_contract_hash=plan.execution_contract_hash,
        base_sha=plan.base_sha,
        expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )
    validate_implementation_approval(approval, plan)
    expired = approval.model_copy(
        update={
            "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        }
    )
    with pytest.raises(ValueError, match="expired"):
        validate_implementation_approval(expired, plan)
    with pytest.raises(ValueError, match="does not match"):
        validate_implementation_approval(
            approval,
            plan.model_copy(update={"plan_markdown": "different"}),
        )
