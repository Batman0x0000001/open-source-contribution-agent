"""
load_settings()
      │
      ├── 读取 .env（如果安装了 dotenv）
      ├── 读取环境变量
      └── 返回 Settings

            ↓

create_anthropic_client(settings)
      │
      ├── 导入 Anthropic SDK
      ├── 处理 Base URL
      └── 创建 Anthropic Client
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in normal use
    load_dotenv = None


@dataclass(frozen=True)#配置对象创建后不可修改
class Settings:
    anthropic_api_key: str | None
    anthropic_base_url: str | None
    model_id: str
    fallback_model_id: str | None
    github_token: str | None = None
    max_agent_rounds: int = 30
    max_total_tokens: int = 200_000
    agent_deadline_seconds: int = 1_800
    repeat_action_limit: int = 3
    consecutive_failure_limit: int = 3
    no_progress_limit: int = 6
    max_changed_files: int = 5
    max_diff_lines: int = 400


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def load_settings() -> Settings:
    """读取环境变量并组装运行配置，测试中可不提供真实 API key。"""
    if load_dotenv is not None:
        load_dotenv(override=True)

    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        github_token=os.getenv("GITHUB_TOKEN"),
        model_id=os.getenv("MODEL_ID", "claude-3-5-sonnet-latest"),
        fallback_model_id=os.getenv("FALLBACK_MODEL_ID") or None,
        max_agent_rounds=_env_int("OSC_AGENT_MAX_ROUNDS", 30),
        max_total_tokens=_env_int("OSC_AGENT_MAX_TOKENS", 200_000),
        agent_deadline_seconds=_env_int("OSC_AGENT_DEADLINE_SECONDS", 1_800),
        repeat_action_limit=_env_int("OSC_AGENT_REPEAT_ACTION_LIMIT", 3),
        consecutive_failure_limit=_env_int("OSC_AGENT_FAILURE_LIMIT", 3),
        no_progress_limit=_env_int("OSC_AGENT_NO_PROGRESS_LIMIT", 6),
        max_changed_files=_env_int("OSC_AGENT_MAX_CHANGED_FILES", 5),
        max_diff_lines=_env_int("OSC_AGENT_MAX_DIFF_LINES", 400),
    )


def create_anthropic_client(settings: Settings):
    """延迟创建 Anthropic client，让单元测试可以直接注入 mock client。"""
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError("Missing dependency: install anthropic to run the CLI.") from exc

    if settings.anthropic_base_url:
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    return Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )
