from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from osc_agent import config
from osc_agent.config import Settings


def test_settings_rejects_legacy_positional_arguments():
    with pytest.raises(TypeError):
        Settings(None, None, "test-model", None)


def test_create_anthropic_client_has_no_environment_side_effect(monkeypatch):
    captured: dict[str, str | int | None] = {}

    class FakeAnthropic:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            max_retries: int,
        ) -> None:
            captured.update(api_key=api_key, base_url=base_url, max_retries=max_retries)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "existing-token")

    client = config.create_anthropic_client(
        Settings(
            anthropic_api_key="api-key",
            anthropic_base_url="https://example.test/anthropic",
        )
    )

    assert isinstance(client, FakeAnthropic)
    assert captured == {
        "api_key": "api-key",
        "base_url": "https://example.test/anthropic",
        "max_retries": 0,
    }
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "existing-token"
