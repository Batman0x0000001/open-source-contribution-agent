"""验证真实模型冒烟流程的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

from osc_agent.application import (
    AgentApplicationConfig,
    AgentProfile,
    UserPrompt,
    build_agent_application,
)
from osc_agent.config import load_settings
from osc_agent.runtime.models import RunCompleted


@pytest.mark.live_model
def test_opt_in_live_model_can_complete_one_runtime_turn(tmp_path: Path) -> None:
    if os.getenv("OSC_AGENT_LIVE_MODEL") != "1":
        pytest.skip("set OSC_AGENT_LIVE_MODEL=1 to run the paid live-model smoke")
    settings = load_settings()
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    if not settings.model_id:
        pytest.skip("MODEL_ID is not configured")
    application = build_agent_application(
        AgentApplicationConfig(
            settings=settings,
            repository_root=tmp_path,
            profile=AgentProfile(
                profile_id="live_smoke",
                system_prompt="Answer directly without tools.",
            ),
        )
    )
    conversation = application.open_session(f"live-model-smoke-{uuid4()}")

    async def run():
        return [
            event
            async for event in conversation.submit(
                UserPrompt(text="Reply with the single word READY.")
            )
        ]

    assert isinstance(asyncio.run(run())[-1], RunCompleted)
