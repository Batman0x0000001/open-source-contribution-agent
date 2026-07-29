"""从已验证的 Bot 合同构造 Agent 的初始 Skill 输入。"""

from __future__ import annotations

from collections.abc import Mapping

from osc_agent.application import SkillInput
from osc_agent.bot.models import BotJob, ExecutionContract, IssuePlanArtifact


def build_planning_skill_input(
    job: BotJob,
    contract: ExecutionContract,
    issue_evidence: Mapping[str, object],
    *,
    skill_name: str,
) -> SkillInput:
    _validate_contract_binding(job, contract)
    return SkillInput(
        name=skill_name,
        arguments={
            "issue_evidence": dict(issue_evidence),
            "base_sha": job.base_sha,
            "execution_contract_hash": job.execution_contract_hash,
        },
    )


def build_implementation_skill_input(
    job: BotJob,
    contract: ExecutionContract,
    plan: IssuePlanArtifact,
    *,
    skill_name: str,
) -> SkillInput:
    _validate_contract_binding(job, contract)
    if (
        plan.base_sha != job.base_sha
        or plan.execution_contract_hash != job.execution_contract_hash
    ):
        raise ValueError("approved plan does not match the Bot Job contract")
    return SkillInput(
        name=skill_name,
        arguments={
            "repo_url": f"https://github.com/{job.repository_full_name}",
            "goal": f"Implement approved plan for issue #{job.issue_number}",
            "mode": "approved_implementation",
            "automation": {
                "issue_number": job.issue_number,
                "base_sha": job.base_sha,
                "approved_plan": plan.plan_markdown,
                "execution_contract_hash": job.execution_contract_hash,
            },
        },
    )


def _validate_contract_binding(job: BotJob, contract: ExecutionContract) -> None:
    if (
        contract.contract_hash != job.execution_contract_hash
        or contract.repository_id != job.repository_id
        or contract.repository_full_name != job.repository_full_name
        or contract.base_sha != job.base_sha
        or contract.issue_input_hash != job.issue_input_hash
    ):
        raise ValueError("Bot Job does not match its execution contract")
