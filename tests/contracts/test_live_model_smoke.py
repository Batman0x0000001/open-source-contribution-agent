"""验证真实模型冒烟流程的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest

from osc_agent.composition import build_application
from osc_agent.config import load_settings
from osc_agent.runtime.models import (
    RunCompleted,
    RuntimeMessage,
    StartQueryParams,
    TextBlock,
)


@pytest.mark.live_model
def test_opt_in_live_model_can_complete_one_runtime_turn(tmp_path: Path) -> None:
    if os.getenv("OSC_AGENT_LIVE_MODEL") != "1":
        pytest.skip("set OSC_AGENT_LIVE_MODEL=1 to run the paid live-model smoke")
    settings = load_settings()
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    if not settings.model_id:
        pytest.skip("MODEL_ID is not configured")
    services = build_application(settings=settings, repo_root=tmp_path)

    async def run():
        return [
            event
            async for event in services.runtime.query(
                StartQueryParams(
                    session_id=f"live-model-smoke-{uuid4()}",
                    model=settings.model_id,
                    system_prompt="Answer directly without tools.",
                    messages=[
                        RuntimeMessage(
                            role="user",
                            content=[TextBlock(text="Reply with the single word READY.")],
                        )
                    ],
                    repository_root=str(tmp_path),
                    config=services.query_config,
                )
            )
        ]

    assert isinstance(asyncio.run(run())[-1], RunCompleted)
