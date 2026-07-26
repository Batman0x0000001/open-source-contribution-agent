from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """进程配置的唯一校验边界；未知环境变量不会进入业务状态。"""

    model_config = SettingsConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, validation_alias="ANTHROPIC_BASE_URL")
    model_id: str | None = Field(default=None, validation_alias="MODEL_ID")
    max_agent_rounds: int = Field(default=30, gt=0, validation_alias="OSC_AGENT_MAX_ROUNDS")
    max_total_tokens: int = Field(default=200_000, gt=0, validation_alias="OSC_AGENT_MAX_TOKENS")
    agent_deadline_seconds: int = Field(default=1_800, ge=0, validation_alias="OSC_AGENT_DEADLINE_SECONDS")
    no_progress_limit: int = Field(default=6, gt=0, validation_alias="OSC_AGENT_NO_PROGRESS_LIMIT")
    model_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias="OSC_AGENT_MODEL_MAX_ATTEMPTS",
    )
    model_retry_base_seconds: float = Field(
        default=1,
        ge=0,
        le=60,
        validation_alias="OSC_AGENT_MODEL_RETRY_BASE_SECONDS",
    )
    model_retry_max_seconds: float = Field(
        default=8,
        ge=0,
        le=300,
        validation_alias="OSC_AGENT_MODEL_RETRY_MAX_SECONDS",
    )
    subprocess_env_allowlist: frozenset[str] = Field(
        default_factory=frozenset,
        validation_alias="OSC_AGENT_SUBPROCESS_ENV_ALLOWLIST",
    )

    def __init__(self, **values: Any) -> None:
        # 配置对象只接受具名字段，避免位置参数被误解释为 Pydantic 的内部选项。
        super().__init__(**values)


def load_settings() -> Settings:
    """读取环境变量并组装运行配置，测试中可不提供真实 API key。"""
    load_dotenv(override=False)

    return Settings()
