from __future__ import annotations

import os

import pytest

from osc_agent.config import create_anthropic_client, load_settings
from osc_agent.harness.stage_agents import run_design_generation, score_candidate_issues


pytestmark = [
    pytest.mark.live_model,
    pytest.mark.skipif(
        os.getenv("OSC_AGENT_RUN_LIVE_TESTS") != "1",
        reason="set OSC_AGENT_RUN_LIVE_TESTS=1 to run live model checks",
    ),
]


def test_live_model_returns_structured_design(tmp_path):
    settings = load_settings()
    client = create_anthropic_client(settings)
    result = run_design_generation(
        client,
        settings,
        {
            "top_directions": [{"name": "Add retry test", "entry": "tests/test_retry.py"}],
            "candidate_issues": [],
            "evidence_pack": {"symbols": {}},
        },
        "Add retry test",
        repo_root=tmp_path,
    )

    assert result is not None
    assert result["allowed_files"]
    assert result["acceptance_checks"]


def test_live_model_returns_explainable_issue_level(tmp_path):
    settings = load_settings()
    client = create_anthropic_client(settings)
    result = score_candidate_issues(
        client,
        settings,
        [
            {
                "number": 1,
                "title": "Add focused retry coverage",
                "body": "Expected retry behavior is missing and can be verified with one unit test.",
            }
        ],
        {1: []},
        repo_root=tmp_path,
    )

    assert result
    assert result[0]["level"] in {"HIGH", "MEDIUM", "LOW", "REJECT"}
    assert set(result[0]["dimensions"]) == {"clarity", "unclaimed", "scope", "testability", "reviewability"}
