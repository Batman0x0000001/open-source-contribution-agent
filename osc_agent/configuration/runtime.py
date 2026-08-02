"""从 YAML 加载 Runtime 轮次、令牌和重试限制。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from osc_agent.configuration.source import load_config_section
from osc_agent.runtime.gateway import RetryPolicy
from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.query_models import QueryConfig


class AgentExecutionConfig(FrozenContractModel):
    max_rounds: int = Field(ge=1, le=100)
    max_total_tokens: int = Field(ge=1)
    deadline_seconds: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_no_progress_rounds: int = Field(ge=1)

    def to_query_config(self) -> QueryConfig:
        return QueryConfig(**self.model_dump())


class AgentConfigs(FrozenContractModel):
    main: AgentExecutionConfig
    explore: AgentExecutionConfig
    verify: AgentExecutionConfig


class ModelRetryConfig(FrozenContractModel):
    max_attempts: int = Field(ge=1, le=10)
    base_seconds: float = Field(ge=0, le=60)
    max_seconds: float = Field(ge=0, le=300)

    @model_validator(mode="after")
    def base_delay_must_not_exceed_maximum(self) -> "ModelRetryConfig":
        if self.base_seconds > self.max_seconds:
            raise ValueError("base_seconds must not exceed max_seconds")
        return self

    def to_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(**self.model_dump())


class RuntimeConfig(FrozenContractModel):
    agents: AgentConfigs
    model_retry: ModelRetryConfig


def default_runtime_config_path() -> Path:
    return Path(__file__).with_name("runtime.yml")


def load_runtime_config(path: Path) -> RuntimeConfig:
    raw = load_config_section(
        path,
        section="runtime",
        source_name="runtime",
    )
    return RuntimeConfig.model_validate(raw)
