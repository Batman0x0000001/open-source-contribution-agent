"""注册并按名称解析代码内置的子 Agent。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from osc_agent.contracts import ContractModel, FrozenContractModel
from osc_agent.subagents.models import SubagentDefinition


SubagentContract = ContractModel | FrozenContractModel


@dataclass(frozen=True)
class SubagentRegistration:
    definition: SubagentDefinition
    input_model: type[SubagentContract]
    output_model: type[SubagentContract]
    prompt_builder: Callable[[SubagentContract], str]
    read_only: bool
    concurrency_safe: bool
    max_parallel: int

    def __post_init__(self) -> None:
        name = self.definition.name
        if name != name.strip():
            raise ValueError("subagent name must be canonical")
        if self.definition.capabilities.allowed_tools is None:
            raise ValueError("subagent capabilities must explicitly enumerate allowed tools")
        contracts = (ContractModel, FrozenContractModel)
        if not issubclass(self.input_model, contracts):
            raise TypeError("subagent input_model must be a strict contract")
        if not issubclass(self.output_model, contracts):
            raise TypeError("subagent output_model must be a strict contract")
        if not isinstance(self.read_only, bool) or not isinstance(self.concurrency_safe, bool):
            raise TypeError("subagent execution flags must be booleans")
        if (
            not isinstance(self.max_parallel, int)
            or isinstance(self.max_parallel, bool)
            or self.max_parallel < 1
        ):
            raise ValueError("subagent max_parallel must be positive")


class SubagentRegistry:
    """只接受应用组装阶段代码注册的内置子 Agent。"""

    def __init__(self, registrations: list[SubagentRegistration] | None = None) -> None:
        self._registrations: dict[str, SubagentRegistration] = {}
        for registration in registrations or []:
            self.register(registration)

    def register(self, registration: SubagentRegistration) -> None:
        name = registration.definition.name
        if name in self._registrations:
            raise ValueError(f"duplicate subagent: {name}")
        self._registrations[name] = registration

    def get(self, name: str) -> SubagentRegistration | None:
        return self._registrations.get(name)

    def list(self) -> list[SubagentRegistration]:
        return [self._registrations[name] for name in sorted(self._registrations)]
