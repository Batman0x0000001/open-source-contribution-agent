from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from osc_agent.runtime_config import (
    RuntimeConfig,
    default_runtime_config_path,
    load_runtime_config,
)


class Settings(BaseSettings):
    """进程配置的唯一校验边界；未知环境变量不会进入业务状态。"""

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
    _runtime_config: RuntimeConfig = PrivateAttr()

    def __init__(self, **values: Any) -> None:
        # 配置对象只接受具名字段，避免位置参数被误解释为 Pydantic 的内部选项。
        super().__init__(**values)
        object.__setattr__(self, "_runtime_config", load_runtime_config(self.runtime_config_path))

    @property
    def runtime(self) -> RuntimeConfig:
        return self._runtime_config


def load_settings() -> Settings:
    """读取环境变量并组装运行配置，测试中可不提供真实 API key。"""
    load_dotenv(override=False)

    return Settings()
