"""提供提交规划产物和交付草稿的 Bot 专用工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import Field

from osc_agent.bot.models import DeliveryDraft, IssuePlanArtifact
from osc_agent.bot.store import BotStore
from osc_agent.runtime.models import ContractModel, ToolError, ToolResult, ToolUseContext
from osc_agent.runtime.tool import BaseTool
from osc_agent.workspaces.git_state import git_workspace_fingerprint


class SubmitIssuePlanInput(IssuePlanArtifact):
    pass


class SubmitIssuePlanOutput(ContractModel):
    artifact_id: str = Field(min_length=1)
    status: str


class SubmitIssuePlanTool(BaseTool[SubmitIssuePlanInput, SubmitIssuePlanOutput]):
    name = "submit_issue_plan"
    description = "Submit the strict final issue plan artifact for the current GitHub bot planning job."
    input_model = SubmitIssuePlanInput
    output_model = SubmitIssuePlanOutput

    def __init__(self, store: BotStore, *, job_id: str, base_sha: str, execution_contract_hash: str = "0" * 64) -> None:
        self.store = store
        self.job_id = job_id
        self.base_sha = base_sha
        self.execution_contract_hash = execution_contract_hash

    def is_read_only(self, input: SubmitIssuePlanInput) -> bool:
        return True

    async def call(self, input: SubmitIssuePlanInput, context: ToolUseContext) -> ToolResult:
        if input.base_sha != self.base_sha:
            return ToolResult(error=ToolError(code="PLAN_BASE_MISMATCH", message="plan base SHA differs from the job base SHA"))
        if input.execution_contract_hash != self.execution_contract_hash:
            return ToolResult(error=ToolError(code="PLAN_CONTRACT_MISMATCH", message="plan execution contract differs from the job"))
        artifact = IssuePlanArtifact.model_validate(input.model_dump(mode="json"))
        artifact_id = await asyncio.to_thread(self.store.save_artifact, self.job_id, artifact)
        return ToolResult(data={"artifact_id": artifact_id, "status": artifact.status})


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

    def __init__(self, store: BotStore, *, job_id: str, issue_number: int, base_sha: str, execution_contract_hash: str = "0" * 64) -> None:
        self.store = store
        self.job_id = job_id
        self.issue_number = issue_number
        self.base_sha = base_sha
        self.execution_contract_hash = execution_contract_hash

    def is_read_only(self, input: SubmitDeliveryDraftInput) -> bool:
        return True

    async def call(self, input: SubmitDeliveryDraftInput, context: ToolUseContext) -> ToolResult:
        fingerprint = await asyncio.to_thread(
            git_workspace_fingerprint, repo_root=Path(context.working_directory)
        )
        if input.issue_number != self.issue_number or input.base_sha != self.base_sha:
            return ToolResult(error=ToolError(code="DELIVERY_JOB_MISMATCH", message="delivery draft does not belong to this job"))
        if input.execution_contract_hash != self.execution_contract_hash:
            return ToolResult(error=ToolError(code="DELIVERY_CONTRACT_MISMATCH", message="delivery draft execution contract differs from the job"))
        if input.snapshot_fingerprint != fingerprint:
            return ToolResult(error=ToolError(code="DELIVERY_FINGERPRINT_MISMATCH", message="delivery draft is not bound to the current workspace"))
        artifact = DeliveryDraft.model_validate(input.model_dump(mode="json"))
        artifact_id = await asyncio.to_thread(self.store.save_artifact, self.job_id, artifact)
        return ToolResult(data={"artifact_id": artifact_id, "workspace_fingerprint": fingerprint})
