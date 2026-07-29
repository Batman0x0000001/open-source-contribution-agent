"""组装本地 CLI 使用的 AgentApplication。"""

from pathlib import Path

from osc_agent.application import (
    AgentApplication,
    AgentApplicationConfig,
    AgentProfile,
    build_agent_application,
)
from osc_agent.application.models import ApprovalHandler, QuestionHandler
from osc_agent.configuration import load_agent_settings


def build_cli_application(
    repository_root: Path,
    profile: AgentProfile,
    *,
    approval_handler: ApprovalHandler,
    question_handler: QuestionHandler,
) -> AgentApplication:
    return build_agent_application(
        AgentApplicationConfig(
            settings=load_agent_settings(),
            repository_root=repository_root,
            profile=profile,
            approval_handler=approval_handler,
            question_handler=question_handler,
        )
    )
