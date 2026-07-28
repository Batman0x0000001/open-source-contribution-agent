"""导出子 Agent 定义、注册表与运行器的公共接口。"""

from osc_agent.agents.definitions import AgentDefinition, AgentInvocation, AgentRunResult
from osc_agent.agents.registry import AgentRegistration, AgentRegistry
from osc_agent.agents.runner import AgentRunner
from osc_agent.agents.verify import VerificationReport, build_verify_registration

__all__ = [
    "AgentDefinition",
    "AgentInvocation",
    "AgentRegistration",
    "AgentRegistry",
    "AgentRunResult",
    "AgentRunner",
    "VerificationReport",
    "build_verify_registration",
]
