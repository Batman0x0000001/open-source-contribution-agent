"""提供提交规划产物和交付草稿的 Bot 专用工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import Field

from osc_agent.bot.domain.artifacts import DeliveryDraft, IssuePlanArtifact
from osc_agent.bot.persistence.store import BotStore
from osc_agent.contracts import ContractModel
from osc_agent.runtime.state import ToolContext
from osc_agent.runtime.tool_models import ToolError, ToolResult
from osc_agent.runtime.tool import BaseTool
from osc_agent.workspaces.git_state import git_workspace_fingerprint


class SubmitIssuePlanInput(ContractModel):
    status: Literal["ready", "blocked"]
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)


class SubmitIssuePlanOutput(ContractModel):
    artifact_id: str = Field(min_length=1)
    status: str


class SubmitIssuePlanTool(BaseTool[SubmitIssuePlanInput, SubmitIssuePlanOutput]):
    name = "submit_issue_plan"
    description = (
        "Finish the current Bot plan. The trusted worker reads the saved plan draft and "
        "binds it to the job; write_plan must be called first."
    )
    input_model = SubmitIssuePlanInput
    output_model = SubmitIssuePlanOutput

    def __init__(self, store: BotStore, *, job_id: str, worker_id: str, base_sha: str, execution_contract_hash: str = "0" * 64) -> None:
        self.store = store
        self.job_id = job_id
        self.worker_id = worker_id
        self.base_sha = base_sha
        self.execution_contract_hash = execution_contract_hash

    def is_read_only(self, input: SubmitIssuePlanInput) -> bool:
        return True

    async def call(self, input: SubmitIssuePlanInput, context: ToolContext) -> ToolResult:
        if input.status == "ready" and input.unresolved_questions:
            return ToolResult(
                error=ToolError(
                    code="PLAN_STATUS_INVALID",
                    message="ready plan cannot contain unresolved questions",
                )
            )
        if input.status == "blocked" and not input.unresolved_questions:
            return ToolResult(
                error=ToolError(
                    code="PLAN_STATUS_INVALID",
                    message="blocked plan requires unresolved questions",
                )
            )
        try:
            plan_markdown = _saved_plan(context)
            summary = _plan_summary(plan_markdown)
            artifact = IssuePlanArtifact(
                status=input.status,
                base_sha=self.base_sha,
                execution_contract_hash=self.execution_contract_hash,
                summary=summary,
                plan_markdown=plan_markdown,
                unresolved_questions=input.unresolved_questions,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return ToolResult(error=ToolError(code="PLAN_DRAFT_INVALID", message=str(exc)))
        artifact_id = await asyncio.to_thread(
            self.store.save_artifact,
            self.job_id,
            artifact,
            required_lease_owner=self.worker_id,
        )
        return ToolResult(data={"artifact_id": artifact_id, "status": artifact.status})


def _saved_plan(context: ToolContext) -> str:
    expected = f"{context.session_id}.md"
    if context.permissions.mode != "plan" or context.permissions.plan_path != expected:
        raise ValueError("current session has no trusted plan draft")
    root = (Path(context.state_directory) / "plans").resolve()
    path = (root / expected).resolve()
    if path.parent != root or not path.is_file():
        raise ValueError("saved plan draft does not exist")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("saved plan draft is empty")
    return content


def _plan_summary(plan_markdown: str) -> str:
    first = next(
        (
            line.strip().lstrip("#").strip()
            for line in plan_markdown.splitlines()
            if line.strip()
        ),
        "",
    )
    return first[:4_000] or "Issue implementation plan"


class SubmitDeliveryDraftInput(DeliveryDraft):
    pass


class SubmitDeliveryDraftOutput(ContractModel):
    artifact_id: str = Field(min_length=1)
    workspace_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class SubmitDeliveryDraftTool(BaseTool[SubmitDeliveryDraftInput, SubmitDeliveryDraftOutput]):
    name = "submit_delivery_draft"
    description = "Submit a strict local delivery draft after tests, Verify, and the final Git snapshot."
    input_model = SubmitDeliveryDraftInput
    output_model = SubmitDeliveryDraftOutput

    def __init__(self, store: BotStore, *, job_id: str, worker_id: str, issue_number: int, base_sha: str, execution_contract_hash: str = "0" * 64) -> None:
        self.store = store
        self.job_id = job_id
        self.worker_id = worker_id
        self.issue_number = issue_number
        self.base_sha = base_sha
        self.execution_contract_hash = execution_contract_hash

    def is_read_only(self, input: SubmitDeliveryDraftInput) -> bool:
        return True

    async def call(self, input: SubmitDeliveryDraftInput, context: ToolContext) -> ToolResult:
        fingerprint = await asyncio.to_thread(
            git_workspace_fingerprint,
            repo_root=Path(context.workspace.working_directory),
        )
        if input.issue_number != self.issue_number or input.base_sha != self.base_sha:
            return ToolResult(error=ToolError(code="DELIVERY_JOB_MISMATCH", message="delivery draft does not belong to this job"))
        if input.execution_contract_hash != self.execution_contract_hash:
            return ToolResult(error=ToolError(code="DELIVERY_CONTRACT_MISMATCH", message="delivery draft execution contract differs from the job"))
        if input.snapshot_fingerprint != fingerprint:
            return ToolResult(error=ToolError(code="DELIVERY_FINGERPRINT_MISMATCH", message="delivery draft is not bound to the current workspace"))
        artifact = DeliveryDraft.model_validate(input.model_dump(mode="json"))
        artifact_id = await asyncio.to_thread(
            self.store.save_artifact,
            self.job_id,
            artifact,
            required_lease_owner=self.worker_id,
        )
        return ToolResult(data={"artifact_id": artifact_id, "workspace_fingerprint": fingerprint})
