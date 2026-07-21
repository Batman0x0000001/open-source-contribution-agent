from __future__ import annotations

import pytest

from osc_agent.harness.command import CommandKind, classify_command


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "python -m unittest tests.test_agent",
        "uv run pytest tests/test_agent.py",
        "npm test",
        "pnpm test",
        "cargo test",
        "go test ./...",
        "dotnet test",
        "./gradlew test",
        "cd src && pytest",
    ],
)
def test_classify_command_recognizes_test_runners(command):
    assert classify_command(command) is CommandKind.TEST


@pytest.mark.parametrize("command", ["echo pytest", "rg unittest", "npm install", "cargo build"])
def test_classify_command_avoids_keyword_false_positives(command):
    assert classify_command(command) is CommandKind.OTHER
