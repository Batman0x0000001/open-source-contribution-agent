"""根据应用配置构建带重试策略的模型 Provider。"""

from __future__ import annotations

from osc_agent.configuration import AgentSettings
from osc_agent.providers.anthropic import AnthropicModelGateway
from osc_agent.runtime.gateway import ModelGateway, RetryingModelGateway


def build_model_gateway(
    settings: AgentSettings,
    injected: ModelGateway | None = None,
) -> ModelGateway:
    if injected is None and not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when no ModelGateway is injected")
    provider = injected or AnthropicModelGateway.from_config(
        api_key=settings.anthropic_api_key or "",
        base_url=settings.anthropic_base_url,
    )
    return RetryingModelGateway(provider, settings.runtime.model_retry.to_retry_policy())
