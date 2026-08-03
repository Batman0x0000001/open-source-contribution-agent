"""子 Agent 公共接口；建议按 models → registry → runner → tool → builtins 阅读。"""

from osc_agent.subagents.models import (
    SubagentDefinition,
    SubagentRequest,
    SubagentRunResult,
)
from osc_agent.subagents.registry import SubagentRegistration, SubagentRegistry
from osc_agent.subagents.runner import SubagentRunner

__all__ = [
    "SubagentDefinition",
    "SubagentRegistration",
    "SubagentRegistry",
    "SubagentRequest",
    "SubagentRunResult",
    "SubagentRunner",
]
