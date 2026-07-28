"""注册并按名称解析可用的子 Agent。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from osc_agent.agents.definitions import AgentDefinition
from osc_agent.runtime.models import ContractModel, FrozenContractModel


AgentContract = ContractModel | FrozenContractModel


@dataclass(frozen=True)
class AgentRegistration:
    definition: AgentDefinition
    input_model: type[AgentContract]
    output_model: type[AgentContract]
    prompt_builder: Callable[[AgentContract], str]
    read_only: bool
    concurrency_safe: bool
    max_parallel: int

    def __post_init__(self) -> None:
        name = self.definition.name
        if name != name.strip():
            raise ValueError("agent name must be canonical")
        contracts = (ContractModel, FrozenContractModel)
        if not issubclass(self.input_model, contracts):
            raise TypeError("agent input_model must be a strict contract")
        if not issubclass(self.output_model, contracts):
            raise TypeError("agent output_model must be a strict contract")
        if not isinstance(self.read_only, bool) or not isinstance(self.concurrency_safe, bool):
            raise TypeError("agent execution flags must be booleans")
        if (
            not isinstance(self.max_parallel, int)
            or isinstance(self.max_parallel, bool)
            or self.max_parallel < 1
        ):
            raise ValueError("agent max_parallel must be positive")


class AgentRegistry:
    """只接受应用组装阶段代码注册的内置 Agent。"""

    def __init__(self, registrations: list[AgentRegistration] | None = None) -> None:
        self._registrations: dict[str, AgentRegistration] = {}
        for registration in registrations or []:
            self.register(registration)

    def register(self, registration: AgentRegistration) -> None:
        name = registration.definition.name
        if name in self._registrations:
            raise ValueError(f"duplicate agent: {name}")
        self._registrations[name] = registration

    def get(self, name: str) -> AgentRegistration | None:
        return self._registrations.get(name)

    def list(self) -> list[AgentRegistration]:
        return [self._registrations[name] for name in sorted(self._registrations)]
