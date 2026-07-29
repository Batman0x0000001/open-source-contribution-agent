"""显式加载 CLI 与 Bot Worker 共用的 Agent 执行配置。"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from osc_agent.configuration.runtime import (
    RuntimeConfig,
    default_runtime_config_path,
    load_runtime_config,
)
from osc_agent.contracts import FrozenContractModel


class _AgentEnvironmentSettings(BaseSettings):
    """只解析环境变量，不执行文件读取。"""

    model_config = SettingsConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, validation_alias="ANTHROPIC_BASE_URL")
    model_id: str | None = Field(default=None, validation_alias="MODEL_ID")
    runtime_config_path: Path = Field(
        default_factory=default_runtime_config_path,
        validation_alias="OSC_AGENT_RUNTIME_CONFIG",
    )
    subprocess_env_allowlist: frozenset[str] = Field(
        default_factory=frozenset,
        validation_alias="OSC_AGENT_SUBPROCESS_ENV_ALLOWLIST",
    )


class AgentSettings(FrozenContractModel):
    """已经解析完成、可直接注入 Application 的 Agent 设置。"""

    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    model_id: str | None = None
    runtime_config_path: Path
    subprocess_env_allowlist: frozenset[str] = frozenset()
    runtime: RuntimeConfig


def load_agent_settings() -> AgentSettings:
    """显式读取环境变量和 Runtime YAML；现有环境变量优先于 .env。"""

    load_dotenv(override=False)
    environment = _AgentEnvironmentSettings()
    return AgentSettings(
        anthropic_api_key=environment.anthropic_api_key,
        anthropic_base_url=environment.anthropic_base_url,
        model_id=environment.model_id,
        runtime_config_path=environment.runtime_config_path,
        subprocess_env_allowlist=environment.subprocess_env_allowlist,
        runtime=load_runtime_config(environment.runtime_config_path),
    )
