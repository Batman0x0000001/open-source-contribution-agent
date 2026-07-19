"""
load_settings()
      │
      ├── 读取 .env
      ├── 读取环境变量
      └── 返回 Settings

            ↓

create_anthropic_client(settings)
      │
      ├── 导入 Anthropic SDK
      ├── 传入 API Key 和 Base URL
      └── 创建 Anthropic Client
"""
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
    model_id: str = Field(default="claude-3-5-sonnet-latest", validation_alias="MODEL_ID")
    fallback_model_id: str | None = Field(default=None, validation_alias="FALLBACK_MODEL_ID")
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")
    max_agent_rounds: int = Field(default=30, gt=0, validation_alias="OSC_AGENT_MAX_ROUNDS")
    max_total_tokens: int = Field(default=200_000, gt=0, validation_alias="OSC_AGENT_MAX_TOKENS")
    agent_deadline_seconds: int = Field(default=1_800, ge=0, validation_alias="OSC_AGENT_DEADLINE_SECONDS")
    repeat_action_limit: int = Field(default=3, gt=0, validation_alias="OSC_AGENT_REPEAT_ACTION_LIMIT")
    consecutive_failure_limit: int = Field(default=3, gt=0, validation_alias="OSC_AGENT_FAILURE_LIMIT")
    no_progress_limit: int = Field(default=6, gt=0, validation_alias="OSC_AGENT_NO_PROGRESS_LIMIT")
    max_changed_files: int = Field(default=5, gt=0, validation_alias="OSC_AGENT_MAX_CHANGED_FILES")
    max_diff_lines: int = Field(default=400, gt=0, validation_alias="OSC_AGENT_MAX_DIFF_LINES")

    def __init__(self, **values: Any) -> None:
        # 配置对象只接受具名字段，避免位置参数被误解释为 Pydantic 的内部选项。
        super().__init__(**values)


def load_settings() -> Settings:
    """读取环境变量并组装运行配置，测试中可不提供真实 API key。"""
    load_dotenv(override=True)

    return Settings()


def create_anthropic_client(settings: Settings):
    """延迟创建 Anthropic client，让单元测试可以直接注入 mock client。"""
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError("Missing dependency: install anthropic to run the CLI.") from exc

    return Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        # 重试、截止时间、Trace 和 fallback 由 harness 统一编排，避免 SDK 双重重试。
        max_retries=0,
    )
