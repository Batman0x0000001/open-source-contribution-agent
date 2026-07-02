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


def load_settings() -> Settings:
    """读取环境变量并组装运行配置，测试中可不提供真实 API key。"""
    if load_dotenv is not None:
        load_dotenv(override=True)

    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model_id=os.getenv("MODEL_ID", "claude-3-5-sonnet-latest"),
        fallback_model_id=os.getenv("FALLBACK_MODEL_ID") or None,
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
