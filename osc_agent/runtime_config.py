from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict, Field, model_validator

from osc_agent.runtime.gateway import RetryPolicy
from osc_agent.runtime.models import FrozenContractModel, QueryConfig


class AgentExecutionConfig(FrozenContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_rounds: int = Field(ge=1, le=100)
    max_total_tokens: int = Field(ge=1)
    deadline_seconds: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_no_progress_rounds: int = Field(ge=1)

    def to_query_config(self) -> QueryConfig:
        return QueryConfig(**self.model_dump())


class AgentConfigs(FrozenContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    main: AgentExecutionConfig
    explore: AgentExecutionConfig
    verify: AgentExecutionConfig


class ModelRetryConfig(FrozenContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
    model_config = ConfigDict(extra="forbid", frozen=True)

    agents: AgentConfigs
    model_retry: ModelRetryConfig


def default_runtime_config_path() -> Path:
    return Path(__file__).with_name("runtime.yml")


def load_runtime_config(path: Path) -> RuntimeConfig:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to read runtime config {path}: {exc}") from exc
    if isinstance(raw, dict) and "runtime" in raw:
        unknown = set(raw) - {"runtime", "repositories"}
        if unknown:
            raise ValueError(f"unknown production config sections: {', '.join(sorted(unknown))}")
        raw = raw["runtime"]
    return RuntimeConfig.model_validate(raw)
