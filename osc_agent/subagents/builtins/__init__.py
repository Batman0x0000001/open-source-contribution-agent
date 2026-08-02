"""导出代码内置的 Explore 与 Verify 子 Agent。"""

from osc_agent.runtime.query_models import QueryConfig
from osc_agent.subagents.builtins.explore import (
    ExploreReport,
    build_explore_subagent,
)
from osc_agent.subagents.builtins.verify import (
    VerificationReport,
    build_verify_subagent,
)
from osc_agent.subagents.registry import SubagentRegistration


def build_default_subagents(
    *,
    model: str,
    explore_config: QueryConfig,
    verify_config: QueryConfig,
) -> tuple[SubagentRegistration, ...]:
    """返回通用 Application 使用的唯一默认子 Agent 集合。"""

    return (
        build_explore_subagent(model=model, config=explore_config),
        build_verify_subagent(model=model, config=verify_config),
    )

__all__ = [
    "ExploreReport",
    "VerificationReport",
    "build_default_subagents",
    "build_explore_subagent",
    "build_verify_subagent",
]
