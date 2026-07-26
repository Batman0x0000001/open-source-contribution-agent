from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from osc_agent.bot.config import load_repository_catalog
from osc_agent.bot.models import (
    BotApproval,
    DeliveryDraft,
    IssuePlanArtifact,
    RepositoryBotConfig,
    plan_evidence_hash,
    validate_implementation_approval,
)


SHA = "a" * 40
FINGERPRINT = "b" * 64


def test_repository_config_requires_a_recognized_test() -> None:
    with pytest.raises(ValidationError, match="recognized test"):
        RepositoryBotConfig(image="worker:latest", validation_commands=("python -m pip check",))
    config = RepositoryBotConfig(
        image="worker:latest",
        validation_commands=("python -m pytest", "python -m pip check"),
    )
    assert config.validation_commands[0] == "python -m pytest"


def test_repository_catalog_is_strict_and_rejects_escape(tmp_path: Path) -> None:
    path = tmp_path / "repositories.yml"
    path.write_text(
        "repositories:\n  owner/repo:\n    image: worker:latest\n    validation_commands: [python -m pytest]\n",
        encoding="utf-8",
    )
    assert "owner/repo" in load_repository_catalog(path).repositories
    with pytest.raises(ValidationError, match="repository-relative"):
        RepositoryBotConfig(
            image="worker:latest",
            validation_commands=("python -m pytest",),
            denied_paths=("../secret",),
        )


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
